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
from modules.active_learning import evaluate_on_test_set, train_committee


##### PARAMETERS #####
NUM_CLASSES = 4
INITIAL_LABELS = 800
BATCH_SIZE = 32
EPOCHS = 5 
QUERY_SIZE = 400
AL_ROUNDS = 5
COMMITTEES = [3,5,9,17] 
TEST_SET_RATIO = 0.1 
######################


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

df = pd.read_csv('data/extracted_dataset.csv')
dataset = DatasetClassifier(df, transform=transform)

committee_accuracies = [[] for _ in COMMITTEES]

for c_i, COMMITTEE_SIZE in enumerate(COMMITTEES):
    print('##############################################')
    print(f'STARTING COMMITTEE {c_i+1} | MEMBER SIZE: {COMMITTEE_SIZE}')
    print('##############################################')

    random.seed(333)
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
    print('----------------------------------------------')

    test_accuracies = []  # Store test set accuracy over rounds

    for round in range(AL_ROUNDS):
        print(f"\n===== Active Learning Round {round+1}/{AL_ROUNDS} =====")
        
        # Train Committee
        committee, committee_accuracy = train_committee(dataset, labeled_indices, COMMITTEE_SIZE, BATCH_SIZE, EPOCHS, device, NUM_CLASSES)

        # Evaluate on held-out test set
        test_accuracy = evaluate_on_test_set(dataset, test_indices, committee, BATCH_SIZE, device)
        test_accuracies.append(test_accuracy)

        # Uncertainty Estimation (Variance-based QBC)
        print("\nEstimating uncertainty...")
        uncertain_samples = []
        unlabeled_subset = Subset(dataset, unlabeled_indices)
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=BATCH_SIZE, shuffle=False)
        
        for model in committee:
            model.eval()
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(unlabeled_loader):
                images = images.to(device)
                committee_outputs = [torch.softmax(model(images), dim=1).cpu().numpy() for model in committee]
                committee_outputs = np.array(committee_outputs)  # Shape: (committee_size, batch_size, num_classes)
                variance = np.var(committee_outputs, axis=0).sum(axis=1)  # Compute variance across models
                uncertain_samples.extend(zip(unlabeled_indices[:len(variance)], variance))

        print(f"Top 5 most uncertain sample variances: {sorted([x[1] for x in uncertain_samples], reverse=True)[:5]}")

        uncertain_samples.sort(key=lambda x: x[1], reverse=True)  # Sort by highest uncertainty
        selected_samples = [x[0] for x in uncertain_samples[:QUERY_SIZE]]
        
        labeled_indices.extend(selected_samples)
        unlabeled_indices = [idx for idx in unlabeled_indices if idx not in selected_samples]
        
        print(f"Added {len(selected_samples)} most uncertain samples to training set.")
        print(f"Total labeled samples: {len(labeled_indices)}, Remaining unlabeled samples: {len(unlabeled_indices)}")

        if not unlabeled_indices:
            print("No more unlabeled data left. Stopping Active Learning.")
            break

    committee_accuracies[c_i].append(test_accuracies)

    print('----------------------------------------------')
    print(f'ENDING COMMITTEE {c_i} | INITIATING NEXT COMMITTEE')
    print('##############################################')

df = pd.DataFrame(
    data=np.array(committee_accuracies).T,
    columns=[f"committee{i+1}" for i in range(len(committee_accuracies))],
    index=[f"round{r+1}" for r in range(len(committee_accuracies[0]))]
)

df.to_csv('AL_results.csv')