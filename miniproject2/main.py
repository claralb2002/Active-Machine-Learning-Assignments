# packages
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import random
import pandas as pd
from torch.utils.data import DataLoader, Subset

# modules
from modules.networks import SimpleResNet
from modules.data_classifier import DatasetClassifier
from modules.dataloader import get_bootstrapped_dataloader
from modules.active_learning import active_learning_loop


##### PARAMETERS #####
NUM_CLASSES = 4
INITIAL_LABELS = 1000
BATCH_SIZE = 32
EPOCHS = 5 
QUERY_SIZE = 500
AL_ROUNDS = 5 
COMMITTEE_SIZE = 3 
TEST_SET_RATIO = 0.1 
######################


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),  # Ensure grayscale is converted to RGB
    transforms.ToTensor(),
])

df = pd.read_csv('data/extracted_dataset.csv')

dataset = DatasetClassifier(df, transform=transform)

num_samples = len(dataset)
unlabeled_indices = list(range(num_samples))
random.shuffle(unlabeled_indices)

test_set_size = int(num_samples * TEST_SET_RATIO)
test_indices = unlabeled_indices[:test_set_size]  # Reserve test samples
unlabeled_indices = unlabeled_indices[test_set_size:]  # Remove test samples from the pool

labeled_indices = unlabeled_indices[:INITIAL_LABELS]
unlabeled_indices = unlabeled_indices[INITIAL_LABELS:]

print(f"Total samples: {num_samples}")
print(f"Test set size (held-out): {len(test_indices)}")
print(f"Initially labeled samples: {len(labeled_indices)}")
print(f"Initially unlabeled samples: {len(unlabeled_indices)}")

labeled_indices, unlabeled_indices, trained_committee, test_accuracies = active_learning_loop(
    dataset, labeled_indices, unlabeled_indices, test_indices, AL_ROUNDS, QUERY_SIZE, COMMITTEE_SIZE, BATCH_SIZE, EPOCHS, device, NUM_CLASSES
)