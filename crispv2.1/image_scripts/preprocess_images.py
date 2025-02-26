from PIL import Image
import cv2
import numpy as np
import os
import argparse
import json
Image.MAX_IMAGE_PIXELS = None

# python image_scripts/preprocess_images.py --image_preprocess_config experiment_configs/image_preprocess.json

def preprocess_images(preprocess_config):

    image_folder = preprocess_config.get("image_folder", "img_input")
    preprocessed_output_folder = preprocess_config.get("preprocessed_output_folder", "preprocessed_images")

    image_size = (224, 224)

    # List all files in the folder
    for filename in os.listdir(image_folder):
        # Check if the file is an image
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Construct the full file path
            file_path = os.path.join(image_folder, filename)
            
            if os.path.exists(file_path):
                print(file_path)
                try:
                    with Image.open(file_path) as img:
                        
                        img_gray = img.convert("L")
                        
                        img_array_gray = np.array(img_gray)

                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # ADAPTIVE HE
                        equalized_img_array = clahe.apply(img_array_gray)

                        equalized_img = Image.fromarray(equalized_img_array)

                        intermediate_size = (equalized_img.size[0] // 4, equalized_img.size[1] // 4)
                        img_intermediate = equalized_img.resize(intermediate_size, Image.Resampling.NEAREST)
                        img_resized = img_intermediate.resize(image_size, Image.Resampling.NEAREST)

                        img_array = np.array(img_resized) / 255.0

                        os.makedirs(preprocessed_output_folder, exist_ok=True)
                        output_image_path = os.path.join(preprocessed_output_folder, f"{os.path.splitext(filename)[0]}.npy")
                        print('Saving', output_image_path)
                        np.save(output_image_path, img_array)

                    del img_intermediate
                    del img_resized
                    del img_array

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRISP Image Preprocessing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Path to config json file
    parser.add_argument('--image_preprocess_config', default='experiment_configs/image_preprocess.json')
    opt = parser.parse_args()

    with open(os.path.join(os.getcwd(), opt.image_preprocess_config)) as json_file:
        preprocess_config = json.load(json_file)

    preprocess_images(preprocess_config)