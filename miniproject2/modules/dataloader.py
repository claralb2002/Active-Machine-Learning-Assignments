from torch.utils.data import DataLoader, Subset
import numpy as np

def get_bootstrapped_dataloader(dataset,indices,batch_size):
    sampled_indices = np.random.choice(indices, size=len(indices), replace=True)
    subset = Subset(dataset, sampled_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)