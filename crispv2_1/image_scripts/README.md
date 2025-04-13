# Steps for running CRISP with image modules

## Package Installation
Ensure you have a working version of **Python 3.7** and then need to install all required packages. 

Conda users, can use the following commands:

`conda create --name crisp2_1 python=3.7`
`conda activate crisp2_1`

Then, use the followiwng command to install packages from the virtual environment:
`pip install -r requirements_new.txt`

## Notes
  - Currently, the image module of CRISP only supports **binary classification (two classes, 0 or 1)**. To run the image modules of the CRISP, first have an image dataset ready in a folder along with each image's class label.

## 1. Image Preprocess
Run the image preprocessing script to resize, normalize, and augment the images:\
`python image_scripts/preprocess_images.py --image_preprocess_config experiment_configs/image_preprocess.json`

Image pre-process configuration JSON would require few parameters
  - `image_folder`: Location of the input images
  - `preprocessed_output_folder`: Path where the preprocessed images will be stored for future use
  - `environments`: Name of the Environment split variable

The script will save a CSV file (e.g., labels.csv) inside the output folder, listing the original and transformed image filenames along with their corresponding environment names (e.g., `iamge_name`, `env_split`) . User can add the image labels and sample/subject keys in this CSV file for running through the the next steps. 

## 2. Train CRISP ensemble of models
Once the images are preprocessed, you can train the CRISP ensemble. The training pipeline now supports both image-only runs and multimodal runs (image + tabular data).

**Run the main training script:**
`python main.py --experiment_config experiment_configs/config.json`

### Configuration File Overview
The configuration file controls the behavior of the pipelines. Following is a sample config JSON to demonstrate the key new sections added for the image model include:

```json
{
    "name": "Example synthetic",
    "short_name": "example",
    "experiment_type": "multimodal", 
    "verbose": 1,
    "test_val_split": [0.1, 0.1],
    "data_options": {
        "dataset_fp": "data/merged_db.pickle",
        "subject_keys": "Subj_ID",
        "targets": ["label"],
        "predictors": "All",
        "environments": ["env_split"],
        "exclude": ["sample", "image_name", "label", "env_split"],
	"output_data_regime": "binary"
    },
    "feature_selection_options": {
        "max_features": 20,
        "verbose": 0,
        "seed": 12
    },
    "ensemble_options": {
        "models": ["RF", "ICP", "NLICP", "IRM", "LIRM"]
    },
    
    "image_data": {
        "image_dir": "data/preprocessed_images/",
        "labels_csv": "data/csvs/rr3_image_dataset_labels.csv",
        "model_type": "DenseNet121",
        "image_model_training_type": "train_test_split", 
        "split_ratio": 0.85,
        "augmentation": true, 
        "batch_size": 64,
        "learning_rate": 0.0001,
        "num_epochs": 100,
        "model_save_path": "image_model_saved/image_model.pth",
        "gradcam_features_save_path": "data/gradcam_features.pkl",
        "tabular_features_path": "data/tabular_db.pickle",

        "image_model_gradcam": {
            "apply_gradcam": true,
            "gradcam_output_save_path": "data/gradcam_outputs"
        },

        "gradcam_features_explainer":{
            "save_path": "data/gradcam_features_explainer",
            "show_clusters": false,
            "show_com": false
        }
    },
    
    "multimodal_merge_options":{
        "environment_split_unified": {
            "env1": {"img_env": ["rotate_90_transform", "gaussian_blur_transform", "brightness_contrast_transform"], "tabular_env": [0]},
            "env2": {"img_env": ["horizontal_flip_transform", "original_resized", "vertical_flip_transform"], "tabular_env": [1]}
        }
    }
}
```

The `experiment_type` parameter is added to distinguish between type of experiment run. It can have value such as `multimodal` for both image and tabular data mix training. For CRISP to use only image data, it can have value `image_only`. For any other value such as `tabular`, it will run CRISP like earlier version with tabular dataset as specified in the `dataset_fp` parameter.

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

**Grad-CAM & Feature Visualization**
- `image_model_gradcam`:
    `apply_gradcam`: Boolean flag to save Grad-CAM heatmaps for all images (the save location is specified via `gradcam_output_save_path`).
- `gradcam_features_explainer`:
    - `save_path`: Folder path for saving Grad-CAM visualizations.
    - `show_clusters`: Boolean flag to visualize cluster contours on the heatmap.
    - `show_com`: Boolean flag to visualize the center of mass on the heatmap.
 
**Image and Tabular Data Merge**
- `multimodal_merge_options`: This parameter takes the environment split names to perform tabular and image merge based on mentioned config. Here is an example if we have 6 environments for image data and 2 envs for tabular data. User can modify as needed:
  - `environment_split_unified`:
    `{
            "env1": {"img_env": ["rotate_90_transform", "gaussian_blur_transform", "brightness_contrast_transform"], "tabular_env": [0]},
            "env2": {"img_env": ["horizontal_flip_transform", "original_resized", "vertical_flip_transform"], "tabular_env": [1]}
        }`
    
    The above example maps as follows:
    Image environments rotate_90_transform, gaussian_blur_transform, brightness_contrast_transform + tabular environment 0 → Unified environment env1.
    Image environments horizontal_flip_transform, original_resized, vertical_flip_transform + tabular environment 1 → Unified environment env2.
