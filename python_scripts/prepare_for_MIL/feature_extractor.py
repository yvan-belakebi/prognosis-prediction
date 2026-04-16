import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens


def load_uni2h_feature_extractor():
    # pretrained=True needed to load UNI2-h weights (and download weights for the first time)
    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
    )
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    model.eval()
    return model, transform


def load_hoptimus1_feature_extractor():
    from torchvision import transforms

    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-1",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False,
    )
    model.to("cuda")
    model.eval()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)
            ),
        ]
    )
    return model, transform


def load_virchow2_feature_extractor():
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model = model.eval()

    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    return model, transform


#### Inference

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
