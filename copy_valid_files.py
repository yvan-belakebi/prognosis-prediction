import os

dest_dir = "WSI/IgA/"
old_dir = "WSI_full/IgA/"

folders = ["coords", "labels", "UNI2-h_feats"]


def extract_basename(file_location):
    if pd.isna(file_location):
        return file_location

    basename_dot_splitted = file_location.split("\\")[-1].split(".")
    if len(basename_dot_splitted) > 1:
        basename_dot_splitted.pop()
    return ".".join(basename_dot_splitted)


import pandas as pd
import shutil

coords_missing = pd.read_csv("missing_coords.csv")
labels_missing = pd.read_csv("missing_labels.csv")
coords_missing_list = coords_missing[coords_missing.columns[0]].tolist()
labels_missing_list = labels_missing[labels_missing.columns[0]].tolist()
for folder in folders:
    for file in os.listdir(os.path.join(old_dir, folder)):
        basename = extract_basename(file)
        if basename in coords_missing_list:
            continue
        if basename in labels_missing_list:
            continue
        dest_path = os.path.join(dest_dir, folder, basename + ".npy")
        file_path = os.path.join(old_dir, folder, file)
        shutil.copy2(file_path, dest_path)
