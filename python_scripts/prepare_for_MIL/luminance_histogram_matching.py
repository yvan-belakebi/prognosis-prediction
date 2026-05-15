from pathlib import Path
import cv2
import numpy as np
from skimage.exposure import match_histograms


def batch_luminance_histogram_match(
    input_dir,
    reference_image_path,
    output_dir,
    image_extensions=("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"),
):
    """
    Match luminance (LAB L-channel) of all images in a folder
    to a reference image.

    Parameters
    ----------
    input_dir : str or Path
        Folder containing source patches.

    reference_image_path : str or Path
        Path to the reference patch/image.

    output_dir : str or Path
        Folder where normalized patches will be saved.

    image_extensions : tuple
        File extensions to process.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load reference image
    # -------------------------
    ref_bgr = cv2.imread(str(reference_image_path))

    if ref_bgr is None:
        raise ValueError(f"Could not load reference image: {reference_image_path}")

    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)
    ref_l = ref_lab[:, :, 0]

    # -------------------------
    # Process all images
    # -------------------------
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(input_dir.glob(ext))

    if len(image_paths) == 0:
        print("No images found.")
        return

    for img_path in image_paths:

        # Load image
        src_bgr = cv2.imread(str(img_path))

        if src_bgr is None:
            print(f"Skipping unreadable file: {img_path}")
            continue

        # Convert to LAB
        src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB)

        # Extract luminance
        src_l = src_lab[:, :, 0]

        # Histogram match luminance only
        matched_l = match_histograms(src_l, ref_l)

        # Replace L channel
        result_lab = src_lab.copy()
        result_lab[:, :, 0] = np.clip(matched_l, 0, 255).astype(np.uint8)

        # Convert back to BGR
        result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

        # Save
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), result_bgr)

        print(f"Processed: {img_path.name}")

    print("Batch luminance histogram matching complete.")
