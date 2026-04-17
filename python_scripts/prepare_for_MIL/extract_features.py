from PIL import Image


def infer_on_image(model, transform, image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(
        dim=0
    )  # Image (torch.Tensor) with shape [1, 3, 224, 224] following image resizing and normalization
    with torch.inference_mode():
        feature_emb = model(
            image
        )  # Extracted features (torch.Tensor) with shape [1,1536] for example
    return feature_emb
