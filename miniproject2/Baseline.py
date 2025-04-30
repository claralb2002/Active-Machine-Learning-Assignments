import torch
import random
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Import necessary modules
from modules.networks import SimpleResNet
from modules.data_classifier import DatasetClassifier
from modules.active_learning import evaluate_on_test_set, train_committee

##### PARAMETERS #####
NUM_CLASSES = 4
INITIAL_LABELS = 600
BATCH_SIZE = 32
EPOCHS = 5 
QUERY_SIZE = 200
AL_ROUNDS = 20
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

random.seed(333)
num_samples = len(dataset)

# Shuffle and split dataset
unlabeled_indices = list(range(num_samples))
random.shuffle(unlabeled_indices)

test_set_size = int(num_samples * TEST_SET_RATIO)
test_indices = unlabeled_indices[:test_set_size]  # Reserve test samples
unlabeled_indices = unlabeled_indices[test_set_size:]  # Remove test samples from the pool
labeled_indices = unlabeled_indices[:INITIAL_LABELS]
unlabeled_indices = unlabeled_indices[INITIAL_LABELS:]

print('##############################################')
print('STARTING BASELINE MODEL - RANDOM SELECTION')
print('##############################################')

print(f"Total samples: {num_samples}")
print(f"Test set size (held-out): {len(test_indices)}")
print(f"Initially labeled samples: {len(labeled_indices)}")
print(f"Initially unlabeled samples: {len(unlabeled_indices)}")
print('----------------------------------------------')

baseline_accuracies = []  # Store test set accuracy over rounds

for round in range(AL_ROUNDS):
    print(f"\n===== Random Selection Round {round+1}/{AL_ROUNDS} =====")
    
    # Train Model (Single model instead of a committee)
    baseline_model, _ = train_committee(dataset, labeled_indices, 1, BATCH_SIZE, EPOCHS, device, NUM_CLASSES)

    # Evaluate on test set
    test_accuracy = evaluate_on_test_set(dataset, test_indices, baseline_model, BATCH_SIZE, device)
    baseline_accuracies.append(test_accuracy)

    # Randomly select new samples
    selected_samples = random.sample(unlabeled_indices, min(QUERY_SIZE, len(unlabeled_indices)))
    
    labeled_indices.extend(selected_samples)
    unlabeled_indices = [idx for idx in unlabeled_indices if idx not in selected_samples]
    
    print(f"Added {len(selected_samples)} random samples to training set.")
    print(f"Total labeled samples: {len(labeled_indices)}, Remaining unlabeled samples: {len(unlabeled_indices)}")

    if not unlabeled_indices:
        print("No more unlabeled data left. Stopping Random Selection.")
        break

print('##############################################')
print('ENDING BASELINE MODEL')
print('##############################################')

# Convert the baseline_accuracies to DataFrame
df_baseline = pd.DataFrame(
    data=baseline_accuracies,  # Data is now directly the accuracies list
    columns=["baseline"],      # The column name is "baseline"
    index=[f"round{r+1}" for r in range(len(baseline_accuracies))]  # Index is round1, round2, etc.
)

# Save to CSV
df_baseline.to_csv('AL_results_baseline.csv')

print("Baseline results saved to AL_results_baseline.csv")
