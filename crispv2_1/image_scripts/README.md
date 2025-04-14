# Steps for running CRISP with image modules

## Package Installation
Ensure to have a working version of **Python 3.7** and then need to install all required packages. The Python version restriction is mainly for supporting the streamlit web output/results, otherwise for only running the experiment scripts future Python versions also work.

Conda users, can use the following commands to create a virtual environment with required Python version:

```sh
conda create --name crisp2_1 python=3.7
```

```sh
conda activate crisp2_1
```

Then, use the following command to install packages from the virtual environment:
```sh
pip install -r requirements_new.txt
```

## Additional Notes
  - CRISP now supports tabular only data (same as previous version), image only, and multi-modal data (tabular + image dataset).
  - Currently, the image module of CRISP only supports **binary classification** (two classes, 0 or 1). **To run the image modules of the CRISP for multimodal or image only experiments, first the user should have an image dataset ready in a directory along with each image's class label.** 
  - The tabular only data analysis modules should work similar to the earlier version of CRISP. The following content only explains `image_only` or `multimodal` experiments.

## 1. Image Preprocess Script
Run the image preprocessing script to resize and augment the images:

```sh
python image_scripts/preprocess_images.py --image_preprocess_config image_preprocess_config_file_name.json
```

Image pre-process configuration JSON would require few fields such as the following example:

```json
{
    "image_folder": "data/img_input",
    "preprocessed_output_folder": "data/preprocessed_images",
    "environments": "env_split"
}
```
  - `image_folder`: Location of the input images
  - `preprocessed_output_folder`: Path where the preprocessed images will be stored for future use during image_only or multimodal training
  - `environments`: Name of the Environment split variable

For the user's convenient, the image pre-processing script will also save a CSV file (e.g., named `labels.csv`) inside the output folder, listing the original and transformed image filenames along with their corresponding environment names (e.g., `image_name`, `env_split`) . User need to then modify this file by adding the image class labels and sample/subject keys (if any) to this CSV file for running through the the next steps during model training. 

The image pre-process module will create the following environments during the image processing and save each image as `.npy` format in corresponding folder. Along with the original (`original_resized`) images there will 5 more environments created by the image pre-processing script. 

The image pre-process module will save images into **6 environments** named as: `original_resized`, `horizontal_flip_transform`, `vertical_flip_transform`, `rotate_90_transform`, `brightness_contrast_transform`, `gaussian_blur_transform`

## 2. Train image model + CRISP ensemble of models
Once the images are preprocessed, the user can train the CRISP ensemble. The training pipeline now supports both image-only experiment and multimodal experiment (image + tabular data).

Sample command to run the main training script:
```sh
python main.py --experiment_config experiment_configs/config_file_name.json
```

### Configuration File Overview
The configuration file controls the behavior of the pipelines. The configuration file has all the fields required by the earlier version of CRISP (Crisp 1.1) and on top of that some new fields are included to accommodate image only and multimodal experiments. 

Following is a sample config JSON to demonstrate the key new sections added for the image model include:

```json
{
    "name": "Example Multimodal",
    "short_name": "multimodal_example",
    "experiment_type": "multimodal", 
    "verbose": 1,
    "test_val_split": [0.2, 0.1],
    "data_options": {
        "dataset_fp": "data/merged_db.pickle",
        "subject_keys": "sample",
        "targets": ["label"],
        "predictors": "All",
        "environments": ["env_split"],
        "exclude": ["sample", "image_name", "label", "env_split"],
	"output_data_regime": "binary"
    },
    "feature_selection_options": {
        "max_features": 40,
        "verbose": 0,
        "seed": 123
    },
    "ensemble_options": {
        "models": ["RF", "ICP", "NLICP", "IRM", "LIRM"]
    },
    
    "image_data": {
        "image_dir": "data/preprocessed_images/",
        "labels_csv": "data/labels.csv",
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

The **`experiment_type`** parameter is added to distinguish between type of experiment run. It can have value such as 
`multimodal`: For experiment both image and tabular data mix
`image_only`: For experiment on image dataset only. Note that, the pickle file name should be the same in both `data_options.dataset_fp` and `image_data.gradcam_features_save_path`

For any other value such as `tabular_only` (this is the default value), it will run CRISP like earlier version with tabular dataset as specified in the `dataset_fp` parameter.

Following are the other newly added fields related to image modules. The required fields are indicated by `*` at the end of each field description. 

**Image Data:** 
  - `image_dir`: Path to the image directory. *
  - `labels_csv`: Path to the image labels CSV file. *
  - `model_type`: Type of model to train (CNN_Scratch or DenseNet121). Default value `DenseNet121`
  - `image_model_training_type`: Type of training/validation (values can be set to `train_test_split` for typical train test with a specified `split_raio`. Another option is `full_loocv` for LOOCV validation based training on full data. Both approaches will save the trained model for later useage such as gradcam features extractions. The `full_loocv` would be more suitable for smaller image dataset, for usual medium/big sized dataset, `train_test_split` should be fine. 
  - `split_ratio`: The training data ratio to train the image model. Default `0.8`
  - `augmentation`: Indicates if image augmentation (rotation, sharpness adjust, resized crop, etc) should be done during image model training. Either `true` or `false`. Default `false`
  - `batch_size`, `learning_rate`, `num_epochs`: Image model's training hyper-parameter. These are optional values for the configuration file. Default values for `batch_size`, `learning_rate`, `num_epochs` are `32`, `0.0001`, and `100`, respectively.
  - `model_save_path`: Path to save the trained model file (extension should be `.pth`). Default value `image_model_saved/image_model.pth`
  - `gradcam_features_save_path`: Path to save the image model's gradcam heatmap features for all the images. (extension should be `.pkl` or `.pickle`) *
  - `tabular_features_path`: Path for the tabular dataset/Gene expression data  (extension should be `.pkl` or `.pickle`) *

**Grad-CAM & Feature Visualization**
- `image_model_gradcam`:
    `apply_gradcam`: Boolean flag (`true` or `false`) indicating whether to save Grad-CAM heatmaps for all images or not (the save location is specified via `gradcam_output_save_path`).
- `gradcam_features_explainer`:
    - `save_path`: Folder path for saving Grad-CAM visualizations.
    - `show_clusters`: Boolean flag (`true` or `false`) indicating whether to save visualizer cluster contours features on the heatmap.
    - `show_com`: Boolean flag (`true` or `false`) indicating whether to save visualizer the center of mass features on the heatmap.
 
**Image and Tabular Data Merge**

The following fields are only required during `multimodal` experiments to combine tabular and image data by an unified setting.

- `multimodal_merge_options`: This parameter takes the environment split names to perform tabular and image merge based on mentioned config. Here is an **example** if we have 6 environments for image data and 2 envs for tabular data. User can modify as needed.
  - `environment_split_unified`:
```json
"environment_split_unified": {
  "env1": {
    "img_env": ["rotate_90_transform", "gaussian_blur_transform", "brightness_contrast_transform"], 
    "tabular_env": [0]
    },
  "env2": {
    "img_env": ["horizontal_flip_transform", "original_resized", "vertical_flip_transform"], 
    "tabular_env": [1]
    }
}
```
    
  The above example maps the image and tabular environments as follows:
  Image environments `rotate_90_transform, gaussian_blur_transform, brightness_contrast_transform` + tabular environment `0` → Unified environment `env1`.

  Image environments `horizontal_flip_transform, original_resized, vertical_flip_transform` + tabular environment `1` → Unified environment `env2`.

The merged dataset pickle file will be saved to the file named in the field `data_options.dataset_fp` so that the CRISP ensemble can use the merged image+tabular features as the dataset.

## 3. Visualise results using streamlit frontend

Run streamlit by running the streamlit_main.py by the following command

```sh
streamlit run streamlit_main.py
```
