import torch

def extract_tumor_features(mask):
    mask = (mask > 0.5).float()
    tumor_area = torch.sum(mask).item()
    mean_intensity = torch.mean(mask).item()
    std_intensity = torch.std(mask).item()
    return [tumor_area, mean_intensity, std_intensity]