# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
# Add the parent directory to sys.path so Python can find 'selection'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time

from functools import partial
from typing import Any, Dict, Optional, Tuple, Union
from warnings import warn

import torch
import torchtune.modules.common_utils as common_utils
from omegaconf import DictConfig, ListConfig, OmegaConf

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchtune import config, modules, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import padded_collate_packed
from torchtune.datasets import ConcatDataset
from torchtune.modules.peft import (
    get_adapter_params,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY

from tqdm import tqdm

from selection import *

# imports from LESS collect_grad_reps.py file
import os
from hashlib import md5
from typing import Dict, Iterable, Optional
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm
from trak.projectors import BasicProjector, CudaProjector, ProjectionType


log = utils.get_logger("DEBUG")


class LoRAFinetuneRecipeSingleDevice(FTRecipeInterface):

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        # Reduced precision logic
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        # fp16 precision is explicitly disabled as it is not supported in this
        # recipe (for example, no gradient scaling).
        if self._dtype == torch.float16:
            raise ValueError(
                "fp16 precision is not supported in this recipe. Please use fp32 or bf16."
            )

        print(cfg)
        print(dir(cfg))
        
        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        if self._log_peak_memory_stats and self._device.type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)

        # activation checkpointing/offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        if self._enable_activation_offloading:
            if self._device.type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif (
            self._enable_activation_checkpointing
            and cfg.checkpointer.model_type != "LLAMA3_VISION"
        ):
            utils.log_rank_zero(
                log,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
       
       
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        if self._resume_from_checkpoint:
            if training.ADAPTER_KEY not in checkpoint_dict:
                raise ValueError(
                    "Adapter weights not found. Please ensure a valid adapter checkpoint is provided."
                )
            # _update_recipe_state will throw an exception if the recipe state is not correctly loaded
            # no need to check here
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        self._metric_logger.log_config(cfg)

        self._compile = cfg.compile
        if cfg.device == "npu" and cfg.compile:
            raise ValueError(
                "NPU does not support model compilation. Please set `compile: False` in the config."
            )
        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)

        # hack to toggle to the low cpu ram version of the reparametrize_as_dtype
        # hook based on the config.
        common_utils._use_low_cpu_ram = cfg.get("low_cpu_ram", False)

        # set up model
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            compile_model=cfg.compile,
            base_model_state_dict=checkpoint_dict[training.MODEL_KEY],
            lora_weights_state_dict=(
                checkpoint_dict[training.ADAPTER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized from file.")

        self._optimizer, self._optimizer_state_dict = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)
        if self._compile:
            self._loss_fn = training.compile_loss(self._loss_fn)

        if self._loss_fn.__class__.__name__ == "CEWithChunkedOutputLoss":
            # set num_output_chunks for model
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)

        log.info("Loss is initialized.")

        # Dataloader depends on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        collate_name = cfg.get("collate_fn", "torchtune.data.padded_collate_sft")
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
        )


        self._steps_per_epoch = (
            # implemented static LESS warmup percentage, so tqdm will regard total bar to desired warmup percentage. 
            int(len(self._dataloader) // self._gradient_accumulation_steps) 
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
            self.global_step = self.epochs_run * self._steps_per_epoch

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # Used to ignore labels for loss computation
        self.ignore_labels_cache = torch.full(
            (cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device
        )

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:

        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        log.info(f" Profiler config after instantiation: {profiler_cfg}")

        self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
        if profiler_cfg["enabled"]:
            self.profiler_wait_steps = profiler_cfg["wait_steps"]
            self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
            self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        compile_model: bool,
        base_model_state_dict: Dict[str, Any],
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)

        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self._lora_attn_modules = list(cfg_model.lora_attn_modules)
        self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
        self._apply_lora_to_output = getattr(cfg_model, "apply_lora_to_output", False)
        self.adapter_params = get_adapter_params(model)
        self._is_dora = any(["magnitude" in k for k in self.adapter_params.keys()])
        set_trainable_params(model, self.adapter_params)

        if compile_model:
            training.compile_model(model)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        base_missing, base_unexpected = model.load_state_dict(
            base_model_state_dict, strict=False
        )
        # This is for any adapters that need to be initialized after base weights
        # have been loaded (e.g. DoRA).
        if self._is_dora:
            for m in model.modules():
                if hasattr(m, "initialize_dora_magnitude"):
                    m.initialize_dora_magnitude()
        if lora_weights_state_dict:
            lora_missing, lora_unexpected = model.load_state_dict(
                lora_weights_state_dict, strict=False
            )
        else:
            lora_missing, lora_unexpected = None, None

        validate_missing_and_unexpected_for_lora(
            lora_attn_modules=self._lora_attn_modules,
            apply_lora_to_mlp=self._apply_lora_to_mlp,
            apply_lora_to_output=self._apply_lora_to_output,
            base_missing=base_missing,
            base_unexpected=base_unexpected,
            lora_missing=lora_missing,
            lora_unexpected=lora_unexpected,
        )
        # Validate model adapter params were loaded in with the expected dtype
        # TODO (rohan-varma): Further validation to ensure the appropriate base params
        # are NF4 vs bf16 based on the quantization config.
        training.validate_expected_param_dtype(
            self.adapter_params.items(), dtype=self._dtype
        )

        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        log.info(f"Model is initialized with precision {self._dtype}.")

        if self._device.type != "cpu":
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)
        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
           print("loading optimizer state dict")
           optimizer.load_state_dict(opt_state_dict)
            

        log.info("Optimizer and loss are initialized.")
        return optimizer, optimizer.state_dict()

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        log.info("Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
    ) -> Tuple[DataLoader]:
        
        
        
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
            packed = False
        else:
            ds = config.instantiate(cfg_dataset, self._tokenizer)
            packed = cfg_dataset.get("packed", False)

        # Instantiate collate_fn
        if "left_pad_sequence" in collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")
        collate_fn = _get_component_from_path(collate_fn)

        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
                if not packed
                else padded_collate_packed
            ),
        )

        log.info("Dataset is initialized.")

        return dataloader

    def _loss_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Shape [b, s], needed for the loss not the model
        labels = batch.pop("labels")
        # run model
        with self.activations_handling_ctx:
            logits = self._model(**batch)

        # Shift labels to compute loss
        # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
        # But this way we dont need to slice the logits. We just add an ignore index to labels.
        labels = torch.hstack(
            (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
        )
        if not isinstance(logits, list):
            labels = labels.reshape(-1)
            logits = logits.reshape(-1, logits.size(-1))

        loss = self._loss_fn(logits, labels)

        # free logits otherwise it peaks backward memory
        del logits

        return loss


    def cleanup(self) -> None:
        self._metric_logger.close()
        
    
    """The following functions are all implemented from the LESS codebase"""
    
    
    def prepare_batch(self, batch, device=torch.device("cuda:0")):
        """ Move the batch to the device. """
        for key in batch:
            batch[key] = batch[key].to(device)


    def get_max_saved_index(self, output_dir: str, prefix="reps") -> int:
        """ 
        Retrieve the highest index for which the data (either representation or gradients) has been stored. 

        Args:
            output_dir (str): The output directory.
            prefix (str, optional): The prefix of the files, [reps | grads]. Defaults to "reps".

        Returns:
            int: The maximum representation index, or -1 if no index is found.
        """

        files = [file for file in os.listdir(output_dir) if file.startswith(prefix)]

        index = [int(file.split(".")[0].split("-")[1]) for file in files]  # e.g., output_dir/reps-100.pt

        return max(index) if len(index) > 0 else -1


    def get_output(self, model,
                   weights: Iterable[Tensor],
                   buffers: Iterable[Tensor],
                   input_ids=None,
                   attention_mask=None,
                   labels=None,
                   ) -> Tensor:

        logits = model(weights, buffers, *(input_ids.unsqueeze(0),
                       attention_mask.unsqueeze(0))).logits
        labels = labels.unsqueeze(0)
        loss_fct = F.cross_entropy
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
        return loss


    def get_trak_projector(self, device: torch.device):
        """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
        try:
            num_sms = torch.cuda.get_device_properties(
                device.index).multi_processor_count
            import fast_jl

            # test run to catch at init time if projection goes through
            fast_jl.project_rademacher_8(torch.zeros(
                8, 1_000, device=device), 512, 0, num_sms)
            projector = CudaProjector
            print("Using CudaProjector")
        except:
            projector = BasicProjector
            print("Using BasicProjector")
        return projector


    def get_number_of_params(self, model):
        """ Make sure that only lora parameters require gradients in peft models. """
        if isinstance(model, PeftModel):
            names = [n for n, p in model.named_parameters() if p.requires_grad and "lora" not in n]

            assert len(names) == 0

        num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])

        print(f"Total number of parameters that require gradients: {num_params}")
        return num_params


    def obtain_gradients(self, model, batch):
        """ obtain gradients. """
        loss = self._loss_step(batch)
        loss.backward()
        vectorized_grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        return vectorized_grads


    # Another method to calulate gradients via LESSSampler.
    # Is not currently utilized. 
    def obtain_sign_gradients(model, batch):
       """ obtain gradients with sign. """
       loss = model(**batch).loss
       loss.backward()
           # Instead of concatenating the gradients, concatenate their signs
       vectorized_grad_signs = torch.cat([torch.sign(p.grad).view(-1) for p in model.parameters() if p.grad is not None])
       return vectorized_grad_signs


    

    def obtain_gradients_with_adam(self, model, batch, avg, avg_sq):
        """ obtain gradients with adam optimizer states. """
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-08

        loss = self._loss_step(batch)
        loss.backward()

        vectorized_grads = torch.cat(
            [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])

        updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
        updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
        vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)

        return vectorized_grads


    def prepare_optimizer_state(self, model, optimizer_state, device):
        # Build mapping from param -> param_id
        param_id_map = {id(p): n for n, p in model.named_parameters()}
        avg_list = []
        avg_sq_list = []

        for param_id, state in optimizer_state["state"].items():
            if "exp_avg" in state and "exp_avg_sq" in state:
                avg_list.append(state["exp_avg"].view(-1))
                avg_sq_list.append(state["exp_avg_sq"].view(-1))
            else:
                name = param_id_map.get(param_id, f"<unknown id {param_id}>")
                print(f"Warning: Missing state entries for param {name}")

        avg = torch.cat(avg_list).to(device)
        avg_sq = torch.cat(avg_sq_list).to(device)

        return avg, avg_sq

    def collect_grads(self, cfg: DictConfig) -> None:
            """
            Collects gradients from the model during evaluation and saves them to disk.

            Args:
                dataloader (torch.utils.data.DataLoader): The data loader for evaluation dataset.
                model (torch.nn.Module): The model from which gradients will be collected.
                output_dir (str): The directory where the gradients will be saved.
                proj_dim List[int]: The dimensions of the target projectors. Each dimension will be saved in a separate folder.
                gradient_type (str): The type of gradients to collect. [adam | sign | sgd]
                adam_optimizer_state (dict): The optimizer state of adam optimizers. If None, the gradients will be collected without considering Adam optimization states. 
                max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
            """

            proj_dim = cfg.gradient_projection_dimension


            model_id = 0  # model_id is used to draft the random seed for the projectors
            block_size = 128  # fixed block size for the projectors
            projector_batch_size = 16  # batch size for the projectors
            torch.random.manual_seed(0)  # set the random seed for torch

            project_interval = 16  # project every 16 batches
            save_interval = 160  # save every 160 batches

            def _project(current_full_grads, projected_grads):
                current_full_grads = torch.stack(current_full_grads).to(torch.float16)
                for i, projector in enumerate(projectors):
                    current_projected_grads = projector.project(
                        current_full_grads, model_id=model_id)
                    projected_grads[proj_dim[i]].append(current_projected_grads.cpu())

            def _save(projected_grads, output_dirs):
                for dim in proj_dim:
                    if len(projected_grads[dim]) == 0:
                        continue
                    projected_grads[dim] = torch.cat(projected_grads[dim])

                    output_dir = output_dirs[dim]
                    outfile = os.path.join(output_dir, f"grads-{count}.pt")
                    torch.save(projected_grads[dim], outfile)
                    print(
                        f"Saving {outfile}, {projected_grads[dim].shape}", flush=True)
                    projected_grads[dim] = []

            device = next(self._model.parameters()).device
            dtype = next(self._model.parameters()).dtype

            # prepare optimization states
            if cfg.info_type == "grads" and cfg.gradient_type == "adam":
                # assert adam_optimizer_state is not None
                # first and second moment estimates
                m, v = self.prepare_optimizer_state(self._model, self._optimizer_state_dict, device)

            projector = self.get_trak_projector(device)
            number_of_params = self.get_number_of_params(self._model)

            # never made it work sadly
            # fmodel, params, buffers = make_functional_with_buffers(model)
            # grads_loss = torch.func.grad(get_output, has_aux=False, argnums=1)

            # initialize a project for each target projector dimension
            projectors = []
            for dim in proj_dim:
                proj = projector(grad_dim=number_of_params,
                                 proj_dim=dim,
                                 seed=0,
                                 proj_type=ProjectionType.rademacher,
                                 device=device,
                                 dtype=dtype,
                                 block_size=block_size,
                                 max_batch_size=projector_batch_size)
                projectors.append(proj)

            count = 0

            # set up a output directory for each dimension
            output_dirs = {}
            for dim in proj_dim:
                output_dir_per_dim = os.path.join(cfg.output_dir, f"dim{dim}")
                output_dirs[dim] = output_dir_per_dim
                os.makedirs(output_dir_per_dim, exist_ok=True)

            # max index for each dimension
            max_index = min(self.get_max_saved_index(
                output_dirs[dim], "grads") for dim in proj_dim)

            # projected_gradients
            full_grads = []  # full gradients
            projected_grads = {dim: [] for dim in proj_dim}  # projected gradients

            for batch in tqdm(self._dataloader, total=len(self._dataloader)):
                self.prepare_batch(batch)
                count += 1

                if count <= max_index:
                    print("skipping count", count)
                    continue
                
                if cfg.gradient_type == "adam":
                    if count == 1:
                        print("Using Adam gradients")

                    vectorized_grads = self.obtain_gradients_with_adam(self._model, batch, m, v)
                    
                elif cfg.gradient_type == "sign":
                    if count == 1:
                        print("Using Sign gradients")
                    vectorized_grads = self.obtain_sign_gradients(self._model, batch)
                else:
                    if count == 1:
                        print("Using SGD gradients")
                    vectorized_grads = self.obtain_gradients(self._model, batch)

                # add the gradients to the full_grads
                full_grads.append(vectorized_grads)
                self._model.zero_grad()

                if count % project_interval == 0:
                    _project(full_grads, projected_grads)
                    full_grads = []

                if count % save_interval == 0:
                    _save(projected_grads, output_dirs)

                if cfg.max_samples is not None and count == cfg.max_samples:
                    break
                
            if len(full_grads) > 0:
                _project(full_grads, projected_grads)
                full_grads = []

            for dim in proj_dim:
                _save(projected_grads, output_dirs)

            torch.cuda.empty_cache()
            for dim in proj_dim:
                output_dir = output_dirs[dim]
                self.merge_and_normalize_info(output_dir, prefix="grads")
                self.merge_info(output_dir, prefix="grads")

            print("Finished")


    def merge_and_normalize_info(output_dir: str, prefix="reps"):
        """ Merge and normalize the representations and gradients into a single file. """
        info = os.listdir(output_dir)
        info = [file for file in info if file.startswith(prefix)]
        # Sort the files in ascending order
        info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
        merged_data = []
        for file in info:
            data = torch.load(os.path.join(output_dir, file))
            normalized_data = normalize(data, dim=1)
            merged_data.append(normalized_data)
        merged_data = torch.cat(merged_data, dim=0)

        output_file = os.path.join(output_dir, f"all_orig.pt")
        torch.save(merged_data, output_file)
        print(
            f"Saving the normalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")



    def merge_info(output_dir: str, prefix="reps"):
        """ Merge the representations and gradients into a single file without normalization. """
        info = os.listdir(output_dir)
        info = [file for file in info if file.startswith(prefix)]
        # Sort the files in ascending order
        info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
        merged_data = []
        for file in info:
            data = torch.load(os.path.join(output_dir, file))
            merged_data.append(data)
        merged_data = torch.cat(merged_data, dim=0)

        output_file = os.path.join(output_dir, f"all_unormalized.pt")
        torch.save(merged_data, output_file)
        print(
            f"Saving the unnormalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")

def recipe_main(cfg: DictConfig = "less/config/llama3_2/step2_gradstore.yaml") -> None:
   
    cfg = OmegaConf.load(cfg)
    config.log_config(recipe_name="LoRAFinetuneRecipeSingleDevice", cfg=cfg)
    recipe = LoRAFinetuneRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.collect_grads(cfg=cfg)

    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())