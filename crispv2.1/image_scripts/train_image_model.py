import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from image_scripts.image_classifier import ImageClassifier, CNN_Scratch 
import json

class ImageDataset(Dataset):
    """ Custom Dataset for Grayscale Image Classification """

    def __init__(self, image_dir, labels_csv, transform=None, augment=False):
        """
        Args:
            image_dir (str): Path to the directory containing images (.png, .jpg, or .npy).
            labels_csv (str): Path to CSV file with image filenames and labels.
            transform (callable, optional): Transform to be applied on an image.
            augment (bool): Whether to apply data augmentation.
        """
        import pandas as pd  # Import here to avoid unnecessary dependencies
        self.image_dir = image_dir
        self.data = pd.read_csv(labels_csv)
        
        # Standard transform for both train and validation sets
        base_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize for model
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Standard normalization for grayscale
        ])
        
        # Augmentations applied only to training data
        if augment:
            print("Applying training image augmentation . . .")
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=10),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                *base_transform.transforms  # Append base transforms
            ])
        else:
            self.transform = base_transform  # Use only base transform

    def __len__(self):
        return len(self.data)

    def load_image(self, img_path):
        """ Load a grayscale image from a file or `.npy` array. """
        if img_path.endswith(".npy"):  # Load .npy file
            image = np.load(img_path)  # Load NumPy array
            if image.ndim != 2:  # Ensure it's grayscale
                raise ValueError(f"Unsupported .npy shape {image.shape} for grayscale image {img_path}")
            image = Image.fromarray(image.astype(np.uint8), mode="L")  # Convert NumPy array to PIL grayscale
        else:
            image = Image.open(img_path).convert("L")  # Load standard images as grayscale

        return image

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.loc[idx, 'image_name'])  # Get image filename
        label = torch.tensor(self.data.loc[idx, 'image_label'], dtype=torch.float32)  # Get label

        image = self.load_image(img_name)  # Load image
        image = self.transform(image)  # Apply transforms

        return image, label
    

def save_model(model, optimizer, model_save_path, save_optimizer=False):
    """ Saves the trained model and optimizer state (if required). """

    # Extract the directory path from the save path
    model_dir = os.path.dirname(model_save_path)

    # Ensure the directory exists
    if not os.path.exists(model_dir):
        print(f"Warning: Directory '{model_dir}' does not exist. Creating it now...")
        os.makedirs(model_dir)  # Create directory

    # Save the model (with or without optimizer)
    save_dict = {"model_state_dict": model.state_dict()}
    if save_optimizer:
        save_dict["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(save_dict, model_save_path)
    print(f"Model saved successfully to {model_save_path}")
    

def train_image_model(config):
    """ Train an image classification model based on config settings """

    # If config is a dictionary, use it directly. Otherwise, load from file.
    if isinstance(config, dict):
        pass  # Use the given dictionary
    else:
        with open(config, "r") as f:
            config = json.load(f)
    
    # Extract paths and parameters
    image_dir = config["image_data"]["image_dir"]
    labels_csv = config["image_data"]["labels_csv"]
    model_type = config["image_data"].get("model_type", "CNN_Scratch")  # Default to CNN_Scratch
    batch_size = config["image_data"].get("batch_size", 16)
    learning_rate = config["image_data"].get("learning_rate", 0.001)
    num_epochs = config["image_data"].get("num_epochs", 10)
    model_save_path = config["image_data"]["model_save_path"]
    split_ratio = config["image_data"].get("split_ratio", 0.8)  # Default: 80% train, 20% val
    augmentation = config["image_data"]["augmentation"]

    # Load dataset
    full_dataset = ImageDataset(image_dir, labels_csv)

    # Split dataset into train and validation
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Wrap train dataset with augmentations
    train_dataset.dataset = ImageDataset(image_dir, labels_csv, augment=augmentation)

    # Create DataLoaders
    full_dataset_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose correct model architecture
    if model_type == "DenseNet121":
        print(f"Running image model {model_type}")
        model = ImageClassifier().to(device)
    else:
        print(f"Running image model {model_type}")
        model = CNN_Scratch().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(torch.device("cuda")), labels.to(torch.device("cuda"))
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(torch.device("cuda")), labels.to(torch.device("cuda"))
                outputs = model(images).squeeze()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        if(config["verbose"]==1):
            print(f"Epoch {epoch+1}/{num_epochs}: Validation Accuracy = {correct / total:.4f}")

    # if(config["verbose"]==1):
    #         print(f"Validation Accuracy = {correct / total:.4f}")

    # Save the trained model
    save_model(model, optimizer, model_save_path)

