# Steps for running CRISP with image modules

## 1. Image Preprocess
Run the image preprocessing script to resize, normalize, and augment the images:\
`python image_scripts/preprocess_images.py --image_preprocess_config experiment_configs/image_preprocess.json`

Image pre-process configuration JSON would require few parameters\
`image_folder`: Location of the input images\
`preprocessed_output_folder`: Path where the preprocessed images will be stored for future use\
`environments`: Name of the Environment split variable

The preprocessing script will also save a CSV (e.g., labels.csv) file listing all the original/transformed filenames and environment numbers.

## 2. Train CRISP ensemble of models\
`python main.py --experiment_config experiment_configs/config.json`

### Configuration File
The configuration file controls the behavior of the pipelines. Key new sections added for the image model include:\

**Image Data:** \
`image_dir`: Path to the image directory.\
`labels_csv`: Path to the image labels CSV file.\
`model_type`: Type of model to train (CNN_Scratch or DenseNet121).\
`image_model_training_type`: Type of training/validation (values can be set to `train_test_split` for typical train test with a specified `split_raio`. Another option is `full_loocv` for LOOCV validation based training on full data. Both approaches will save the trained model for later useage such as gradcam features extractions.\
`gradcam_features_explainer`: Controls visualization of clusters and center of mass.\
