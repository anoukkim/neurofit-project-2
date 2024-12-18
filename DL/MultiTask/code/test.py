import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from utils import extract_tumor_features  

with open("config.json", "r") as f:
    config = json.load(f)

def evaluate_model(multi_input_cnn, unet_model, test_loader, device, output_dir="evaluation_results"):
    os.makedirs(output_dir, exist_ok=True)

    multi_input_cnn.eval()
    unet_model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, masks, features, labels in test_loader:
            images, masks, features, labels = images.to(device), masks.to(device), features.to(device), labels.to(device)

            # Use UNet for mask prediction
            predicted_masks = unet_model(images)

            # Extract features from predicted masks and convert them to Tensors
            mask_features = torch.stack(
                [torch.tensor(extract_tumor_features(mask), dtype=torch.float32, device=device) for mask in predicted_masks]
            )
            combined_features = torch.cat((features, mask_features), dim=1)

            # Use MultiInputCNN for classification
            outputs = multi_input_cnn(images, combined_features)

            # Collect true labels and predictions
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend((outputs > 0.5).float().cpu().numpy())

    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Classification report
    class_report = classification_report(true_labels, predicted_labels, target_names=['Alive', 'Dead'])

    # Save the confusion matrix plot
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Alive', 'Dead'], yticklabels=['Alive', 'Dead'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # Save classification report as a text file
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Classification Report:\n")
        f.write(class_report)

    print(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")
    print(f"Classification report saved to {report_path}")

    return conf_matrix