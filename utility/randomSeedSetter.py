import random
import numpy as np
import torch

def seed_everything(seed=42):
    """Locks all random seeds for absolute reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)