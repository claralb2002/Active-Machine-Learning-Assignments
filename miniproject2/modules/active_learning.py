import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, Subset
from modules.networks import SimpleResNet
from modules.dataloader import get_bootstrapped_dataloader

def evaluate_on_test_set(dataset, test_indices, committee, batch_size, device):
    """ Evaluates the trained committee on the held-out test set after each AL round. """
    print("\nEvaluating Committee on Held-Out Test Set...")

    test_subset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    total_correct = 0
    total_samples = 0

    for model in committee:
        model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Get predictions from all committee members
            committee_outputs = [model(images) for model in committee]
            avg_output = torch.mean(torch.stack(committee_outputs), dim=0)  # Average over committee

            # Compute final predictions
            _, predicted = torch.max(avg_output, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    test_accuracy = (total_correct / total_samples) * 100
    print(f"Test Set Accuracy: {test_accuracy:.2f}%")
    return test_accuracy


def train_committee(dataset, labeled_indices, committee_size, batch_size, epochs, device, num_classes):
    """ Trains a committee of models on the labeled dataset and returns the trained models with final accuracy. """
    committee = [SimpleResNet(num_classes=num_classes).to(device) for _ in range(committee_size)]
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