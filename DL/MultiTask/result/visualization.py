import json
from models import UNet, MultiInputCNN
from dataset import ImageFeatureDataset
from train import train_model
from test import evaluate_model
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from utils import extract_tumor_features

with open("config.json", "r") as f:
    config = json.load(f)

device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_path = config["data"]["base_path"]
csv_file = config["data"]["csv_file"]
combined_df = pd.read_csv(csv_file)

filenames = combined_df['FileName'].tolist()
features = combined_df.iloc[:, 2:-1].drop(columns=['tumor_tissue_site']).to_numpy()
labels = combined_df['death01'].to_numpy()

train_filenames, temp_filenames, train_features, temp_features, train_labels, temp_labels = train_test_split(
    filenames, features, labels, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"]
)
val_filenames, test_filenames, val_features, test_features, val_labels, test_labels = train_test_split(
    temp_filenames, temp_features, temp_labels, test_size=config["data"]["val_split"], random_state=config["data"]["random_state"]
)

transform = None
train_dataset = ImageFeatureDataset(base_path, train_features, train_filenames, train_labels, transform=transform)
val_dataset = ImageFeatureDataset(base_path, val_features, val_filenames, val_labels, transform=transform)
test_dataset = ImageFeatureDataset(base_path, test_features, test_filenames, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["data"]["batch_size"], shuffle=False)

unet_model = UNet().to(device)
unet_model.load_state_dict(torch.load(config["model"]["unet_weights"]))
unet_model.eval()

multi_input_cnn = MultiInputCNN().to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(multi_input_cnn.parameters(), lr=config["training"]["learning_rate"])

multi_input_cnn = MultiInputCNN().to(device)
multi_input_cnn.load_state_dict(torch.load(config["model"]["multi_input_cnn_weights"]))
multi_input_cnn.eval()

print("Starting evaluation...")
results = []
multi_input_cnn.eval()
unet_model.eval()

with torch.no_grad():
    for images, masks, features, labels in test_loader:
        images, masks, features, labels = images.to(device), masks.to(device), features.to(device), labels.to(device)
        predicted_masks = unet_model(images)
        mask_features = torch.stack(
            [torch.tensor(extract_tumor_features(mask), dtype=torch.float32, device=device) for mask in predicted_masks]
        )
        combined_features = torch.cat((features, mask_features), dim=1)
        outputs = multi_input_cnn(images, combined_features)
        predicted_probs = outputs.cpu().numpy()
        predicted_labels = (outputs > 0.5).float().cpu().numpy()
        for i in range(images.size(0)):
            results.append({
                "image": images[i].cpu().squeeze().numpy(),
                "ground_truth_mask": masks[i].cpu().squeeze().numpy(),
                "predicted_mask": predicted_masks[i].cpu().squeeze().numpy(),
                "ground_truth_label": labels[i].cpu().item(),
                "predicted_prob": predicted_probs[i].item(),
                "predicted_label": predicted_labels[i].item()
            })

def save_visualizations(results, output_dir="visualization_results", threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    for idx, result in enumerate(results[10:30]):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(result["image"], cmap='gray')
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        axs[1].imshow(result["ground_truth_mask"], cmap='gray')
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")
        binary_predicted_mask = (result["predicted_mask"] > threshold).astype(np.uint8)
        axs[2].imshow(binary_predicted_mask, cmap='gray')
        axs[2].set_title("Predicted Mask (Binary)")
        axs[2].axis("off")
        plt.suptitle(f"Ground Truth: {'Survived' if result['ground_truth_label'] == 0 else 'Death'} | "
                     f"Predicted: {result['predicted_prob']:.2f} ({'Survived' if result['predicted_label'] == 0 else 'Death'})")
        output_path = os.path.join(output_dir, f"result_{idx}.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved visualization: {output_path}")

save_visualizations(results)
