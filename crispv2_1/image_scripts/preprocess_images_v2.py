import os
import numpy as np
import cv2
from PIL import Image
import albumentations as A

Image.MAX_IMAGE_PIXELS = None

class Transformations:
    """Class containing modular image transformations and preprocessing pipeline."""
    
    def __init__(self):
        self.transforms = {
            "horizontal_flip": A.HorizontalFlip(p=1.0),
            "vertical_flip": A.VerticalFlip(p=1.0),
            "rotate_90": A.Rotate(limit=(90, 90), p=1.0),
            "brightness_contrast": A.RandomBrightnessContrast(p=1.0),
            "gaussian_blur": A.GaussianBlur(blur_limit=(3, 7), p=1.0), 
        }

    def apply_transform(self, image, transform_name):
        """Apply a specific transformation to an image."""
        transform = self.transforms.get(transform_name)
        if transform:
            augmented = transform(image=image)
            return augmented["image"]
        else:
            raise ValueError(f"Transformation '{transform_name}' not found!")
    
    def preprocess_images(self, image_folder):
        """Preprocess images by applying transformations and saving outputs."""
        image_size = (224, 224)
        output_root_folder = f"{image_folder}_processed"
        os.makedirs(output_root_folder, exist_ok=True)
        
        for filename in os.listdir(image_folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(image_folder, filename)
                
                try:
                    with Image.open(file_path) as img:
                        img = np.array(img.convert("RGB"))
                        
                        for transform_name in self.transforms.keys():
                            transformed_img = self.apply_transform(img, transform_name)
                            transformed_img_resized = cv2.resize(transformed_img, image_size)
                            transformed_img_array = transformed_img_resized / 255.0
                            
                            transform_folder = os.path.join(output_root_folder, transform_name)
                            os.makedirs(transform_folder, exist_ok=True)
                            
                            output_image_path = os.path.join(transform_folder, f"{os.path.splitext(filename)[0]}.npy")
                            np.save(output_image_path, transformed_img_array)
                            print(f"Saved: {output_image_path}")
                
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
