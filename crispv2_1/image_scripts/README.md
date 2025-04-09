# Steps for running CRISP with image modules

First need to install all required packages. Use the followiwng command to install packages:
`pip install -r requirements_new.txt`

## 1. Image Preprocess
Run the image preprocessing script to resize, normalize, and augment the images:\
`python image_scripts/preprocess_images.py --image_preprocess_config experiment_configs/image_preprocess.json`

Image pre-process configuration JSON would require few parameters
  - `image_folder`: Location of the input images
  - `preprocessed_output_folder`: Path where the preprocessed images will be stored for future use
  - `environments`: Name of the Environment split variable

The script will save a CSV file (e.g., labels.csv) listing the original and transformed filenames along with their corresponding environment numbers.

## 2. Train CRISP ensemble of models
Once the images are preprocessed, you can train the CRISP ensemble. The training pipeline now supports both image-only runs and multimodal runs (image + tabular data).

**Run the main training script:**
`python main.py --experiment_config experiment_configs/config.json`

### Configuration File Overview
The configuration file controls the behavior of the pipelines. Key new sections added for the image model include:

The `experiment_type` parameter is added to distinguish between type of run. It can have value such as `multimodal` for both image and tabular data mix training. For CRISP to use only image data, it can have value `image_only`. For any other value such as `tabular`, it will run CRISP like earlier version with tabular dataset as specified in the `dataset_fp` parameter.

**Image Data:** 
  - `image_dir`: Path to the image directory.
  - `labels_csv`: Path to the image labels CSV file.
  - `model_type`: Type of model to train (CNN_Scratch or DenseNet121).
  - `image_model_training_type`: Type of training/validation (values can be set to `train_test_split` for typical train test with a specified `split_raio`. Another option is `full_loocv` for LOOCV validation based training on full data. Both approaches will save the trained model for later useage such as gradcam features extractions.
  - `split_ratio`: The training data ratio to train the image model. Default `0.8`
  - `augmentation`: Indicates if image augmentation (rotation, sharpness adjust, resized crop, etc) should be done during image model training. Either `true` or `false`.
  - `batch_size`, `learning_rate`, `num_epochs`: Image model's training hyper-parameter. These are optional values for the configuration file.
  - `model_save_path`: Path to save the trained model file.
  - `gradcam_features_save_path`: Path to save the image model's gradcam heatmap features for all the images.
  - `tabular_features_path`: Path for the prepraed tabular dataset (e.g., Gene expression data)

***Grad-CAM & Feature Visualization***
- `image_model_gradcam`:
    `apply_gradcam`: Boolean flag to save Grad-CAM heatmaps for all images (the save location is specified via `gradcam_output_save_path`).
- `gradcam_features_explainer`:
    - `save_path`: Folder path for saving Grad-CAM visualizations.
    - `show_clusters`: Boolean flag to visualize cluster contours on the heatmap.
    - `show_com`: Boolean flag to visualize the center of mass on the heatmap.
