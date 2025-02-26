import numpy as np
import pandas as pd

def save_merged_features(config):
    tabular_data_df = pd.read_pickle(config["image_data"]["tabular_features_path"])
    image_features_df = pd.read_pickle(config["image_data"]["image_features_save_path"])
    print("Tabular", tabular_data_df.shape, "Image features", image_features_df.shape)

    print("Merging image and tabular data features . . .")
    mered_dataset_save_path = config["data_options"]["dataset_fp"]
    merged_features_df = pd.merge(tabular_data_df, image_features_df, on="sample", how="inner")
    
    merged_features_df.to_csv("data/rr3_dataset/merged_data.csv", index=False)
    merged_features_df.reset_index(drop=True).to_pickle(mered_dataset_save_path)
