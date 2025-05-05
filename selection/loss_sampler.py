import copy
import torch
from selection.selectivesampler import SelectiveSampler


class LossSampler(SelectiveSampler):
    """
    The LossSampler maintains an internal loss buffer and, at the end of the scoring pass,
    computes per-sample selection probabilities and updates its internal mask.
    A copy of the loss_fn is used (with reduction='none') so that the regular
    training loss function is unaffected.
    """

    def __init__(
        self,
        dataset,
        loss_fn: torch.nn.Module,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        sampling_ratio: float = 0.5,
    ):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed
        )
        self._per_sample_loss_fn = copy.deepcopy(loss_fn)
        if hasattr(self._per_sample_loss_fn, "reduction"):
            self._per_sample_loss_fn.reduction = "none"
        self.sampling_ratio = sampling_ratio
        # loss_buffer is a dict mapping dataset index -> accumulated loss and token count.
        self._loss_buffer = {}  # {sample_id: (loss_sum, valid_tokens)}
        self.mask = None  # Boolean tensor of size len(dataset)
        self.no_grad_scoring = True

    def pre_epoch(self) -> None:
        # Reset the loss buffer and mask at the start of each epoch.
        self._loss_buffer = {}
        n = len(self.dataset)
        mask = [True] * n
        self.set_mask(mask)

    def post_epoch(self) -> None:
        pass

    def on_batch(self, idx: int, batch: dict) -> None:
        # No action here; we use inform_logits() for scoring.
        pass

    def after_forward(self, idx: int, batch: dict, current_loss: float) -> None:
        pass

    def inform_logits(
        self, sample_ids: list[int], logits: torch.Tensor, shifted_labels: torch.Tensor
    ) -> None:
        """
        For each sample in the batch compute its per-token losses and aggregate.
        If a sample occurs in multiple sequences, accumulate loss_sum and token_count.
        logits: [B, seq_len, vocab_size]
        shifted_labels: [B, seq_len]
        """
        token_losses = self._per_sample_loss_fn(
            logits, shifted_labels
        ).detach()  # [B, seq_len]
        valid_mask = (
            shifted_labels != self._per_sample_loss_fn.ignore_index
        ).float()  # [B, seq_len]
        loss_sum_batch = (token_losses * valid_mask).sum(dim=1)  # [B]
        token_count_batch = valid_mask.sum(dim=1) + 1e-6  # [B] (avoids divison by zero)
        # For each sample in the batch, update or initialize the aggregation.
        for sid, loss_val, count in zip(
            sample_ids, loss_sum_batch.tolist(), token_count_batch.tolist()
        ):
            if sid in self._loss_buffer:
                prev_loss, prev_count = self._loss_buffer[sid]
                self._loss_buffer[sid] = (prev_loss + loss_val, prev_count + count)
            else:
                self._loss_buffer[sid] = (loss_val, count)

    def sample(self) -> None:
        """
        After the scoring pass, compute average loss per sample for all samples in _loss_buffer,
        compute selection probabilities, and update self.mask (Boolean tensor over dataset).
        """
        if len(self._loss_buffer) == 0:
            print("No samples informed; selecting all samples.")
            self.mask = torch.ones(len(self.dataset), dtype=torch.bool)
            return

        print(f"Saw {len(self._loss_buffer.keys())} sample IDs in total.")

        # Compute average loss for each scored sample.
        all_sample_ids = list(self._loss_buffer.keys())
        avg_losses = []
        for sid in all_sample_ids:
            loss_sum, token_count = self._loss_buffer[sid]
            avg_losses.append(loss_sum / token_count)
        losses = torch.tensor(avg_losses, dtype=torch.float)
        eps = 1e-6
        probs = losses + eps
        probs = probs / probs.sum()

        num_scored = len(all_sample_ids)
        self._num_selected_samples = max(1, int(self.sampling_ratio * num_scored))

        print(
            f"num_scored = {num_scored}, num_select = {self._num_selected_samples}, mask sum = {sum(self.mask)}"
        )

        selected_idxs_in_scored = torch.multinomial(
            probs, self._num_selected_samples, replacement=False
        )
        selected_sample_ids = set(
            all_sample_ids[i] for i in selected_idxs_in_scored.tolist()
        )

        mask = [sid in selected_sample_ids for sid in range(len(self.dataset))]
        assert len(mask) == len(self.mask)
        self.set_mask(mask)

        print(f"new mask sum = {sum(self.mask)}")
