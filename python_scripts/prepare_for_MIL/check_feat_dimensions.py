import pandas as pd
import os


def check_feats_shape(feats_dir, patches_dir, backbone):
    expected_dim = (
        1536
        if backbone == "UNI2-h" or backbone == "h-optimus-1"
        else 1280 if backbone == "Virchow2" else None
    )

    for feat_file in os.listdir(feats_dir):
        if feat_file.endswith(".csv"):
            feat_path = os.path.join(feats_dir, feat_file)
            feats = pd.read_csv(feat_path)
            n_feats, dim = feats.shape

            patch_dir = os.path.join(patches_dir, os.path.splitext(feat_file)[0])
            n_patches = len(
                [
                    f
                    for f in os.listdir(patch_dir)
                    if os.path.isfile(os.path.join(patch_dir, f))
                ]
            )

            assert (
                n_feats == n_patches
            ), f"Number of features {n_feats} does not match number of patches {n_patches} for {feat_file}"
            assert (
                dim == expected_dim + 1
            ), f"Feature dimension {dim} does not match expected {expected_dim} for {feat_file}"

    print("All feature files have the correct shape and number of features.")


if __name__ == "__main__":
    feats_dir = "datasets/prognosis/single/single/"
    patches_dir = "WSI/prognosis/single/"
    check_feats_shape(feats_dir, patches_dir, backbone="UNI2-h")
