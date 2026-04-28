import os
import torch
import timm
from torchvision import transforms

############################################
# 1. UNI2-h
############################################


def load_uni2h_feature_extractor():
    uni_dir = "./models/uni2h"

    uni_kwargs = {
        "model_name": "vit_giant_patch14_224",
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

    uni_model = timm.create_model(pretrained=False, **uni_kwargs)

    uni_weights = torch.load(
        os.path.join(uni_dir, "pytorch_model.bin"), map_location="cpu"
    )

    uni_model.load_state_dict(uni_weights, strict=True)
    uni_model.eval()

    # UNI2-h transform
    transform_uni = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return uni_model, transform_uni


############################################
# 2. VIRCHOW2
############################################


def load_virchow2_feature_extractor():
    virchow_dir = "./models/virchow2"

    virchow_model = timm.create_model(
        "vit_large_patch14_224", pretrained=False, num_classes=0
    )

    virchow_weights = torch.load(
        os.path.join(virchow_dir, "pytorch_model.bin"), map_location="cpu"
    )

    virchow_model.load_state_dict(virchow_weights, strict=False)
    virchow_model.eval()

    # Virchow2 transform
    transform_virchow = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return virchow_model, transform_virchow


############################################
# 3. H-OPTIMUS-1
############################################


def load_hoptimus1_feature_extractor():
    hopt_dir = "./models/hoptimus1"

    hoptimus_model = timm.create_model(
        "vit_large_patch14_224",
        pretrained=False,
        num_classes=0,
        init_values=1e-5,
        dynamic_img_size=False,
    )

    hopt_weights = torch.load(
        os.path.join(hopt_dir, "pytorch_model.bin"), map_location="cpu"
    )

    hoptimus_model.load_state_dict(hopt_weights, strict=False)
    hoptimus_model.eval()

    # H-optimus-1 transform
    transform_hoptimus = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)
            ),
        ]
    )
    return hoptimus_model, transform_hoptimus
