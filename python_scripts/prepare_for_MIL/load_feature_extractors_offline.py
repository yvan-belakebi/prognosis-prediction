import os
import torch
import timm
from timm.models.vision_transformer import VisionTransformer
from torchvision import transforms

############################################
# 1. UNI2-h
############################################


def load_uni2h_feature_extractor():
    uni_dir = "./models/uni2h"

    # Instantiate directly to avoid timm's pretrained-weights registry check
    uni_model = VisionTransformer(
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )

    weights_path = os.path.join(uni_dir, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(uni_dir, "model.safetensors")
    uni_weights = torch.load(weights_path, map_location="cpu")

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

    # Instantiate directly to avoid timm's pretrained-weights registry check.
    # Virchow2 is a ViT-Huge/14 with SwiGLU MLP, SiLU activation and 4 reg tokens.
    virchow_model = VisionTransformer(
        img_size=224,
        patch_size=14,
        depth=32,
        num_heads=16,
        init_values=1e-5,
        embed_dim=1280,
        mlp_ratio=5.3375,
        num_classes=0,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=4,
        dynamic_img_size=True,
    )

    weights_path = os.path.join(virchow_dir, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(virchow_dir, "model.safetensors")
    virchow_weights = torch.load(weights_path, map_location="cpu")

    virchow_model.load_state_dict(virchow_weights, strict=True)
    virchow_model.eval()

    # Virchow2 transform
    transform_virchow = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return virchow_model, transform_virchow


############################################
# 3. H-OPTIMUS-1
############################################


def load_hoptimus1_feature_extractor():
    hopt_dir = "./models/hoptimus1"

    hoptimus_model = timm.create_model(
        "vit_giant_patch14_reg4_dinov2",
        pretrained=False,
        img_size=224,
        num_classes=0,
        init_values=1e-5,
        dynamic_img_size=False,
    )

    weights_path = os.path.join(hopt_dir, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(hopt_dir, "model.safetensors")
    hopt_weights = torch.load(weights_path, map_location="cpu")

    hoptimus_model.load_state_dict(hopt_weights, strict=True)
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
