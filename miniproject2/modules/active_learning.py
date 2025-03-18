import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, Subset
from modules.networks import SimpleResNet
from modules.dataloader import get_bootstrapped_dataloader

def train_committee(dataset, labeled_indices, committee_size, batch_size, epochs, device):
    """ Trains a committee of models on the labeled dataset and returns the trained models with final accuracy. """
    committee = [SimpleResNet(num_classes=NUM_CLASSES).to(device) for _ in range(committee_size)]
    print(f"Initialized {committee_size} committee models.")

    final_accuracies = []
    for model_idx, model in enumerate(committee):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        dataloader = get_bootstrapped_dataloader(dataset, labeled_indices, batch_size)

        print(f"Training Model {model_idx+1}/{committee_size}...")
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # Compute Accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            accuracy = (correct / total) * 100
            print(f"  Model {model_idx+1} - Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Evaluate Final Accuracy of Committee Model
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        final_accuracy = (total_correct / total_samples) * 100
        final_accuracies.append(final_accuracy)

    avg_committee_accuracy = np.mean(final_accuracies)
    print(f"\nFinal Committee Accuracy: {avg_committee_accuracy:.2f}%")
    return committee, avg_committee_accuracy


def active_learning_loop(dataset, labeled_indices, unlabeled_indices, al_rounds, query_size, committee_size, batch_size, epochs, device):
    """ Runs Active Learning for a given number of rounds, training a committee and selecting uncertain samples. """
    for round in range(al_rounds):
        print(f"\n===== Active Learning Round {round+1}/{al_rounds} =====")
        
        # Train Committee
        committee, committee_accuracy = train_committee(dataset, labeled_indices, committee_size, batch_size, epochs, device)

        # Uncertainty Estimation (Variance-based QBC)
        print("\nEstimating uncertainty...")
        uncertain_samples = []
        unlabeled_subset = Subset(dataset, unlabeled_indices)
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=batch_size, shuffle=False)
        
        for model in committee:
            model.eval()
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(unlabeled_loader):
                images = images.to(device)
                committee_outputs = [torch.softmax(model(images), dim=1).cpu().numpy() for model in committee]
                committee_outputs = np.array(committee_outputs)  # Shape: (committee_size, batch_size, num_classes)
                variance = np.var(committee_outputs, axis=0).sum(axis=1)  # Compute variance across models
                uncertain_samples.extend(zip(unlabeled_indices[:len(variance)], variance))

        # Print top uncertain samples
        print(f"Top 5 most uncertain sample variances: {sorted([x[1] for x in uncertain_samples], reverse=True)[:5]}")

        # Select most uncertain samples
        uncertain_samples.sort(key=lambda x: x[1], reverse=True)  # Sort by highest uncertainty
        selected_samples = [x[0] for x in uncertain_samples[:query_size]]
        
        # Update labeled and unlabeled indices
        labeled_indices.extend(selected_samples)
        unlabeled_indices = [idx for idx in unlabeled_indices if idx not in selected_samples]
        
        print(f"Added {len(selected_samples)} most uncertain samples to training set.")
        print(f"Total labeled samples: {len(labeled_indices)}, Remaining unlabeled samples: {len(unlabeled_indices)}")

        if not unlabeled_indices:
            print("No more unlabeled data left. Stopping Active Learning.")
            break
    
    return labeled_indices, unlabeled_indices  # Return updated indices for further analysis
