import json
from models import UNet, MultiInputCNN
from dataset import ImageFeatureDataset
from train import train_model
from test import evaluate_model
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Check for GPU
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
base_path = config["data"]["base_path"]
csv_file = config["data"]["csv_file"]
combined_df = pd.read_csv(csv_file)

# Extract filenames, features, and labels
filenames = combined_df['FileName'].tolist()
features = combined_df.iloc[:, 2:-1].drop(columns=['tumor_tissue_site']).to_numpy()
labels = combined_df['death01'].to_numpy()

# Split data
train_filenames, temp_filenames, train_features, temp_features, train_labels, temp_labels = train_test_split(
    filenames, features, labels, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"]
)
val_filenames, test_filenames, val_features, test_features, val_labels, test_labels = train_test_split(
    temp_filenames, temp_features, temp_labels, test_size=config["data"]["val_split"], random_state=config["data"]["random_state"]
)

# Initialize datasets
transform = None  # Add transformations if needed based on config
train_dataset = ImageFeatureDataset(base_path, train_features, train_filenames, train_labels, transform=transform)
val_dataset = ImageFeatureDataset(base_path, val_features, val_filenames, val_labels, transform=transform)
test_dataset = ImageFeatureDataset(base_path, test_features, test_filenames, test_labels, transform=transform)

# Initialize data loaders
train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["data"]["batch_size"], shuffle=False)

# Initialize models
unet_model = UNet().to(device)
unet_model.load_state_dict(torch.load(config["model"]["unet_weights"]))
unet_model.eval()

multi_input_cnn = MultiInputCNN().to(device)

# Define loss function and optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(multi_input_cnn.parameters(), lr=config["training"]["learning_rate"])

# Train the model
print("Starting training...")
train_model(
    unet_model,
    multi_input_cnn,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    config["training"]["num_epochs"]
)

# Evaluate the model
multi_input_cnn = MultiInputCNN().to(device)
multi_input_cnn.load_state_dict(torch.load(config["model"]["multi_input_cnn_weights"]))
multi_input_cnn.eval()
                                           
print("Starting evaluation...")
conf_matrix = evaluate_model(
    multi_input_cnn,
    unet_model,
    test_loader,
    device
)

# Display confusion matrix
print("Confusion Matrix:")
print(conf_matrix)