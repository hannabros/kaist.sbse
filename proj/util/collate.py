import torch
from collections import defaultdict

def customCollate(batches):
    return [{key: torch.stack(value) for key, value in batch.items()} for batch in batches]