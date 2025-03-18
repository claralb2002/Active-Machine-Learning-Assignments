# packages
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random
from torch.utils.data import DataLoader, Subset
import pandas as pd

# modules
from modules.networks import SimpleResNet
from modules.data_class import DatasetClassifier


##### PARAMETERS #####
NUM_CLASSES = 4
INITIAL_LABELS = 100
BATCH_SIZE = 32
EPOCHS = 5 
QUERY_SIZE = 50 
AL_ROUNDS = 5 
COMMITTEE_SIZE = 3 
######################


transform = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor(),
])

df = pd.read_csv('data/extracted_dataset.csv')

dataset = DatasetClassifier(df,transform=transform)

num_samples = len(dataset)
unlabeled_indices = list(range(num_samples))
random.shuffle(unlabeled_indices)

labeled_indices = unlabeled_indices[:INITIAL_LABELS]
unlabeled_indices = unlabeled_indices[INITIAL_LABELS:]