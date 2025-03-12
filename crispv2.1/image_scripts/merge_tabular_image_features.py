import numpy as np
import pandas as pd

def save_merged_features(config):
    tabular_data_df = pd.read_pickle(config["image_data"]["tabular_features_path"])
    gradcam_features_df = pd.read_pickle(config["image_data"]["gradcam_features_save_path"])
    
    subject_key = config["data_options"]["subject_keys"]
    targets = config["data_options"]["targets"]
    environments = config["data_options"]["environments"]
    
    print("Tabular", tabular_data_df.shape, "GradCAM features", gradcam_features_df.shape)
    # Find overlapping columns
    overlap_columns = tabular_data_df.columns.intersection(gradcam_features_df.columns).tolist()
    if(subject_key in overlap_columns):
        overlap_columns.remove(subject_key)
    print("overlap_columns", overlap_columns)
    # Removing the columns
    tabular_data_df = tabular_data_df.drop(columns=overlap_columns)

    print("Tabular", tabular_data_df.shape, "GradCAM features", gradcam_features_df.shape)

    mered_dataset_save_path = config["data_options"]["dataset_fp"]
    # print("Merging image and tabular data features . . .")
    merged_features_df = pd.merge(gradcam_features_df, tabular_data_df, on="sample", how="inner")
    # merged_features_df = gradcam_features_df.copy()
    
    print("Merged data", merged_features_df.shape)
    print('Saving', mered_dataset_save_path.split('.')[0]+'.csv')
    merged_features_df.to_csv(mered_dataset_save_path.split('.')[0]+'.csv', index=False)
    merged_features_df.reset_index(drop=True).to_pickle(mered_dataset_save_path)
