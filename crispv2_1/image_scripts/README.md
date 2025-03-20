# Steps for running CRISP with image modules

## 1. Image Preprocess
Run the image preprocessing script to resize, normalize, and augment the images:\
`python image_scripts/preprocess_images.py --image_preprocess_config experiment_configs/image_preprocess.json`

Image pre-process configuration JSON would require few parameters:\
`image_folder`: Location of the input images\
`preprocessed_output_folder`: Path where the preprocessed images will be stored for future use\
`environments`: Name of the Environment split variable
