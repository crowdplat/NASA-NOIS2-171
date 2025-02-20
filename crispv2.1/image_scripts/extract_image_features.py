import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import os
from image_scripts.image_classifier import CNN_Scratch, ImageClassifier

def extract_image_features(config):
    """ Extract CNN feature embeddings from images and save as tabular data """

    # Extract parameters from config
    model_save_path = config["image_data"]["model_save_path"]
    image_dir = config["image_data"]["image_dir"]
    labels_csv = config["image_data"]["labels_csv"]
    image_features_save_path = config["image_data"]["image_features_save_path"]
    model_type = config["image_data"].get("model_type", "CNN_Scratch")  # Default to CNN_Scratch

    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose correct model architecture
    if model_type == "DenseNet121":
        model = ImageClassifier().to(device)
    else:
        model = CNN_Scratch().to(device)  # Default to CNN_Scratch

    # Load trained model weights
    checkpoint = torch.load(model_save_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Identify layers of model
    print("Model architecture:")
    print(model)

    
    # Fix: Extract features correctly based on model type
    if model_type == "DenseNet121":
        feature_extractor = model.model.features  # Use only convolutional feature extractor
    else:
        # Modify model to remove fully connected layers for CNN_Scratch
        conv_layers = list(model.children())  # Get all model layers
        fc_start_index = next(
            (i for i, layer in enumerate(conv_layers) if isinstance(layer, torch.nn.Linear)),
            len(conv_layers)  # If no Linear layer is found, keep all layers
        )
        feature_extractor = torch.nn.Sequential(*conv_layers[:fc_start_index])  # Keep only convolutional layers

    print("\nFixed Feature Extractor Architecture:")
    print(feature_extractor)


    # Define image transformation (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


    def load_image(image_path):
        """Loads an image from file (.png, .jpg) or `.npy` array (NumPy)."""
        if image_path.endswith(".npy"):
            image = np.load(image_path)  # Load NumPy array
            if image.ndim != 2:  # Ensure it's grayscale
                raise ValueError(f"Invalid shape {image.shape} for grayscale image {image_path}")
            return Image.fromarray(image.astype(np.uint8), mode="L")  # Convert to PIL grayscale
        else:
            return Image.open(image_path).convert("L")  # Load regular image file


    def extract_features(image_path):
        """Extracts CNN feature vector from an image or `.npy` file."""
        image = load_image(image_path)  # Load correctly based on file type
        image = transform(image).unsqueeze(0).to(device)  # Apply transforms
        # print("image", image.shape)

        with torch.no_grad():
            features = feature_extractor(image)  # Extract features
            
        # Fix: Ensure the feature vector is correctly flattened
        # features = features.view(features.size(0), -1)  # Flatten tensor
        features = torch.flatten(features, start_dim=1)  # Fix flattening
        return features.cpu().numpy().flatten()  # Convert to NumPy 1D vector
    
    
    # Read dataset CSV (Load image filenames + labels)
    df = pd.read_csv(labels_csv)

    # Extract features for each image
    feature_list = []
    for idx, row in df.iterrows():
        # img_filename = row.iloc[0]  # First column: image filename
        # label = row.iloc[1]  # Second column: label
        
        sample = row['sample']  # 1st column: sample name
        img_filename = row['image_name']  # 2nd column: image filename
        label = row['image_label']  # 3rd column: label
        img_path = os.path.join(image_dir, img_filename)

        if os.path.exists(img_path):
            features = extract_features(img_path)
            feature_list.append([sample, img_filename, label] + features.tolist())  # Include label in table
        else:
            print(f"Warning: Image {img_filename} not found. Skipping.")

    # Convert extracted features into a DataFrame
    non_image_feature_columns = ["sample", "image_name", "image_label"]
    columns = non_image_feature_columns + [f"feature_{i}" for i in range(len(feature_list[0]) - len(non_image_feature_columns))]
    features_df = pd.DataFrame(feature_list, columns=columns)

    # features_df['env_split'] = 'env1'

    # Save extracted features to CSV/pickle
    os.makedirs(os.path.dirname(image_features_save_path), exist_ok=True)  # Ensure directory exists
    print("Saving image features . . .", features_df.shape)
    features_df.to_csv("data/rr3_dataset/image_features.csv", index=False)
    features_df.reset_index(drop=True).to_pickle(image_features_save_path)
    print(f"Extracted features saved to {image_features_save_path}")
