from typing import Any
import torch

def collate_with_sample_id(batch: list[dict[str, Any]], base_collate_fn, **kwargs) -> dict[str, Any]:
    """
    Calls the base collate function, then adds a 'sample_ids' field.
    """
    collated = base_collate_fn(batch, **kwargs)
    assert "sample_id" in batch[0]
    sample_ids = [sample["sample_id"] for sample in batch]
    collated["sample_ids"] = torch.tensor(sample_ids, dtype=torch.long)
    return collated
