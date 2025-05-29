import torch

# Load the adapter weights (LoRA only)
adapter = torch.load("model_cache/finetuned_models/Llama-3.2-1B-Instruct/FullSampler_2/epoch_0/adapter_model.pt", map_location="cpu")

total_params = 0
for name, param in adapter.items():
    num_params = param.numel()
    print(f"{name}: {num_params} parameters")
    total_params += num_params

print(f"\nTotal trainable parameters in LoRA adapter: {total_params}")
