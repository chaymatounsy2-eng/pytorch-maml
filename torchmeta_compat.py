"""
Compatibility layer pour remplacer torchmeta avec PyTorch 2.x
"""
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class BatchMetaDataLoader(DataLoader):
    """
    DataLoader compatible MAML pour PyTorch 2.x
    """
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0, **kwargs):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )
        self.batch_size = batch_size

def gradient_update_parameters(model, grads, step_size=0.001, first_order=False):
    """
    Mettre à jour paramètres du modèle avec gradients
    """
    if not isinstance(grads, (list, tuple)):
        grads = [grads]
    
    with torch.no_grad():
        for param, grad in zip(model.parameters(), grads):
            if grad is not None:
                param.copy_(param - step_size * grad)
    
    return model
