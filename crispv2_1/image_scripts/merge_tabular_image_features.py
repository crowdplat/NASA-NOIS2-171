import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def normalize_except_exclude(tabular_data_df, exclude_vars):
    """
    Normalize all columns in the DataFrame except the columns specified in exclude_vars.
    
    Parameters:
    - tabular_data_df: pandas DataFrame containing the data
    - exclude_vars: List of column names to exclude from normalization
    
    Returns:
    - A pandas DataFrame with normalized data except for the excluded columns
    """
    print("exclude_vars", exclude_vars)
    # Separate the columns to exclude from the rest
    columns_to_exclude = tabular_data_df[exclude_vars]
    columns_to_normalize = tabular_data_df.drop(columns=exclude_vars)

    # Normalize the remaining columns using MinMaxScaler
    scaler = MinMaxScaler()  # Use StandardScaler() for standardization instead
    normalized_columns = scaler.fit_transform(columns_to_normalize)

    # Create a DataFrame from the normalized data
    normalized_columns_df = pd.DataFrame(normalized_columns, columns=columns_to_normalize.columns)

    # Concatenate the excluded columns with the normalized columns
    result_df = pd.concat([columns_to_exclude.reset_index(drop=True), normalized_columns_df.reset_index(drop=True)], axis=1)

    # Re-order columns to match the original DataFrame
    result_df = result_df[tabular_data_df.columns]

    return result_df

def save_merged_features(config):
    tabular_data_df = pd.read_pickle(config["image_data"]["tabular_features_path"])
    gradcam_features_df = pd.read_pickle(config["image_data"]["gradcam_features_save_path"])
    
    subject_key = config["data_options"]["subject_keys"]
    target_var = config["data_options"]["targets"]
    environments = config["data_options"]["environments"]
    exclude = config["data_options"]["exclude"]
    
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

    exclude_vars = list(set(exclude+list(target_var)+list(environments)+list(['num_activation_clusters'])))
    merged_features_df = normalize_except_exclude(merged_features_df, exclude_vars)
    
    print("Merged data", merged_features_df.shape)
    print('Saving', mered_dataset_save_path.split('.')[0]+'.csv')
    merged_features_df.to_csv(mered_dataset_save_path.split('.')[0]+'.csv', index=False)
    merged_features_df.reset_index(drop=True).to_pickle(mered_dataset_save_path)
