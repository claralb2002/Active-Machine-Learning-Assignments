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
from modules.data_class import DatasetClassifier
from modules.dataloader import get_bootstrapped_dataloader


##### PARAMETERS #####
NUM_CLASSES = 4
INITIAL_LABELS = 100
BATCH_SIZE = 32
EPOCHS = 5 
QUERY_SIZE = 50 
AL_ROUNDS = 5 
COMMITTEE_SIZE = 3 
######################

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backend.mps.is_available() else "cpu")

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

# ACTIVE LEARNING LOOP
for round in range(AL_ROUNDS):
    print(f"Active Learning Round {round+1}/{AL_ROUNDS}")
    
    # Train Committee with Bootstrapping
    committee = [SimpleResNet(num_classes=NUM_CLASSES).to(device) for _ in range(COMMITTEE_SIZE)]
    committee_predictions = []
    
    for model in committee:
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        dataloader = get_bootstrapped_dataloader(dataset,labeled_indices,BATCH_SIZE)
        
        for epoch in range(EPOCHS):
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
    # Uncertainty Estimation (Variance-based QBC)
    uncertain_samples = []
    unlabeled_subset = Subset(dataset, unlabeled_indices)
    unlabeled_loader = DataLoader(unlabeled_subset, batch_size=BATCH_SIZE, shuffle=False)
    for model in committee:
        model.eval()
    with torch.no_grad():
        for images, _ in unlabeled_loader:
            images = images.to(device)
            committee_outputs = [torch.softmax(model(images.to(device)), dim=1).cpu().numpy() for model in committee]
            committee_outputs = np.array(committee_outputs)  # Shape: (committee_size, batch_size, num_classes)
            variance = np.var(committee_outputs, axis=0).sum(axis=1)  # Compute variance across models
            uncertain_samples.extend(zip(unlabeled_indices[:len(variance)], variance))
    
    # Select most uncertain samples
    uncertain_samples.sort(key=lambda x: x[1], reverse=True)  # Sort by highest uncertainty
    selected_samples = [x[0] for x in uncertain_samples[:QUERY_SIZE]]
    
    # Update labeled and unlabeled indices
    labeled_indices.extend(selected_samples)
    unlabeled_indices = [idx for idx in unlabeled_indices if idx not in selected_samples]
    
    print(f"Added {len(selected_samples)} most uncertain samples to training set.")
    
    if not unlabeled_indices:
        print("No more unlabeled data left. Stopping Active Learning.")
        break
