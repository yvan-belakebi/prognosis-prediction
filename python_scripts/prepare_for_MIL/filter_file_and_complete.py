import os
import pandas as pd

coords_dir = "WSI/IgA/UNI2-h_feats/"
labels_dir = "WSI/IgA/labels/"
coords_files = os.listdir(coords_dir)
labels_files = os.listdir(labels_dir)


def extract_basename(file_location):
    if pd.isna(file_location):
        return file_location

    basename_dot_splitted = file_location.split("\\")[-1].split(".")
    basename_dot_splitted.pop()
    return ".".join(basename_dot_splitted)


def rename_to_complete(incomplete_filename, complete_files_list, incomplete_dir):
    incomplete_basename = extract_basename(incomplete_filename)
    for complete_filename in complete_files_list:
        if (
            len(incomplete_basename) >= 7
            and " " in incomplete_basename
            and incomplete_basename in complete_filename
            and complete_filename.count(".") > incomplete_filename.count(".")
        ):
            os.rename(
                os.path.join(incomplete_dir, incomplete_filename),
                os.path.join(incomplete_dir, complete_filename),
            )
            return complete_filename
    return None


changed_coords = [rename_to_complete(c, labels_files, coords_dir) for c in coords_files]
changed_labels = [rename_to_complete(l, coords_files, labels_dir) for l in labels_files]

coords_files_new = os.listdir(coords_dir)
labels_files_new = os.listdir(labels_dir)

coords_files = set(coords_files_new)
labels_files = set(labels_files_new)

missing_coords = labels_files - coords_files
missing_labels = coords_files - labels_files

df = pd.DataFrame(list(missing_coords), columns=["Items"])
df[df.columns[0]] = df[df.columns[0]].apply(extract_basename)
df.to_csv("missing_coords.csv", index=False)
df = pd.DataFrame(list(missing_labels), columns=["Items"])
df[df.columns[0]] = df[df.columns[0]].apply(extract_basename)
df.to_csv("missing_labels.csv", index=False)
