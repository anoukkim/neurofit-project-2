import torch
from tqdm import tqdm
import json
from utils import extract_tumor_features

with open("config.json", "r") as f:
    config = json.load(f)

def train_model(unet_model, multi_input_cnn, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        multi_input_cnn.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for images, masks, features, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            images, masks, features, labels = images.to(device), masks.to(device), features.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                predicted_masks = unet_model(images)

            mask_features = torch.stack(
                [torch.tensor(extract_tumor_features(mask), dtype=torch.float32, device=device) for mask in predicted_masks]
                )
            combined_features = torch.cat((features, mask_features), dim=1)

            outputs = multi_input_cnn(images, combined_features)
            loss = criterion(outputs, labels.unsqueeze(1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct_predictions += ((outputs > 0.5).float() == labels.unsqueeze(1)).sum().item()
            total_predictions += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            torch.save(multi_input_cnn.state_dict(), config["model"]["multi_input_cnn_weights"])
            print("Saved best model weights!")