import dsmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob, copy
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.utils import shuffle
from load_feature_extractor import (
    load_hoptimus1_feature_extractor,
    load_uni2h_feature_extractor,
    load_virchow2_feature_extractor,
)


class BagDataset:
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = Image.open(temp_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"input": img}


class ToTensor(object):
    def __call__(self, img):
        return VF.to_tensor(img)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def bag_dataset(args, csv_file_path, transform=None):
    transformed_dataset = BagDataset(
        csv_file=csv_file_path,
        transform=transform if transform is not None else Compose([ToTensor()]),
    )
    dataloader = DataLoader(
        transformed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return dataloader, len(transformed_dataset)


def compute_feats(
    args,
    bags_list,
    i_classifier,
    save_path=None,
    magnification="single",
    transform=None,
):
    i_classifier.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i_classifier = i_classifier.to(device)
    num_bags = len(bags_list)
    for i in range(0, num_bags):
        feats_list = []
        if magnification == "single" or magnification == "low":
            csv_file_path = glob.glob(os.path.join(bags_list[i], "*.jpg")) + glob.glob(
                os.path.join(bags_list[i], "*.jpeg")
            )
        elif magnification == "high":
            csv_file_path = glob.glob(
                os.path.join(bags_list[i], "*" + os.sep + "*.jpg")
            ) + glob.glob(os.path.join(bags_list[i], "*" + os.sep + "*.jpeg"))
            print()
        dataloader, bag_size = bag_dataset(args, csv_file_path, transform=transform)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch["input"].float().to(device)
                outputs = i_classifier(patches)
                feats = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                # Handle 3D tensor output from transformer models (batch, sequence, features)
                if len(feats.shape) == 3:
                    # Flatten the sequence dimension: (batch, sequence, features) -> (batch * sequence, features)
                    feats = feats.reshape(-1, feats.shape[-1])
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
                sys.stdout.write(
                    "\r Computed: {}/{} -- {}/{}".format(
                        i + 1, num_bags, iteration + 1, len(dataloader)
                    )
                )
        if len(feats_list) == 0:
            print("No valid patch extracted from: " + bags_list[i])
        else:
            df = pd.DataFrame(feats_list)
            os.makedirs(
                os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]),
                exist_ok=True,
            )
            df.to_csv(
                os.path.join(
                    save_path,
                    bags_list[i].split(os.path.sep)[-2],
                    bags_list[i].split(os.path.sep)[-1] + ".csv",
                ),
                index=False,
                float_format="%.4f",
            )


def compute_tree_feats(args, bags_list, embedder_low, embedder_high, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder_low = embedder_low.to(device)
    embedder_high = embedder_high.to(device)
    embedder_low.eval()
    embedder_high.eval()
    num_bags = len(bags_list)
    with torch.no_grad():
        for i in range(0, num_bags):
            low_patches = glob.glob(os.path.join(bags_list[i], "*.jpg")) + glob.glob(
                os.path.join(bags_list[i], "*.jpeg")
            )
            feats_list = []
            feats_tree_list = []
            dataloader, bag_size = bag_dataset(args, low_patches)
            for iteration, batch in enumerate(dataloader):
                patches = batch["input"].float().to(device)
                feats, classes = embedder_low(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
            for idx, low_patch in enumerate(low_patches):
                high_folder = (
                    os.path.dirname(low_patch)
                    + os.sep
                    + os.path.splitext(os.path.basename(low_patch))[0]
                )
                high_patches = glob.glob(high_folder + os.sep + "*.jpg") + glob.glob(
                    high_folder + os.sep + "*.jpeg"
                )
                if len(high_patches) == 0:
                    pass
                else:
                    for high_patch in high_patches:
                        img = Image.open(high_patch).convert("RGB")
                        img = VF.to_tensor(img).float().to(device)
                        feats, classes = embedder_high(img[None, :])

                        if args.tree_fusion == "fusion":
                            feats = feats.cpu().numpy() + 0.25 * feats_list[idx]
                        elif args.tree_fusion == "cat":
                            feats = np.concatenate(
                                (feats.cpu().numpy(), feats_list[idx][None, :]), axis=-1
                            )
                        else:
                            raise NotImplementedError(
                                f"{args.tree_fusion} is not an excepted option for --tree_fusion. This argument accepts 2 options: 'fusion' and 'cat'."
                            )

                        feats_tree_list.extend(feats)
                sys.stdout.write(
                    "\r Computed: {}/{} -- {}/{}".format(
                        i + 1, num_bags, idx + 1, len(low_patches)
                    )
                )
            if len(feats_tree_list) == 0:
                print("No valid patch extracted from: " + bags_list[i])
            else:
                df = pd.DataFrame(feats_tree_list)
                os.makedirs(
                    os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]),
                    exist_ok=True,
                )
                df.to_csv(
                    os.path.join(
                        save_path,
                        bags_list[i].split(os.path.sep)[-2],
                        bags_list[i].split(os.path.sep)[-1] + ".csv",
                    ),
                    index=False,
                    float_format="%.4f",
                )
            print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute features from different backbones and save them as csv files for MIL training."
    )
    parser.add_argument(
        "--num_classes", default=2, type=int, help="Number of output classes [2]"
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="Batch size of dataloader [128]"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of threads for datalodaer"
    )
    parser.add_argument(
        "--gpu_index", type=int, nargs="+", default=(0,), help="GPU ID(s) [0]"
    )
    parser.add_argument(
        "--backbone",
        default="Virchow2",
        type=str,
        help="Embedder backbone [resnet18|resnet34|resnet50|resnet101|h-optimus-1|UNI2-h|Virchow2]",
    )
    parser.add_argument(
        "--norm_layer",
        default="instance",
        type=str,
        help="Normalization layer [instance]",
    )
    parser.add_argument(
        "--magnification",
        default="single",
        type=str,
        help="Magnification to compute features. Use `tree` for multiple magnifications. Use `high` if patches are cropped for multiple resolution and only process higher level, `low` for only processing lower level.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        type=str,
        help="Folder of the pretrained weights, simclr/runs/*",
    )
    parser.add_argument(
        "--weights_high",
        default=None,
        type=str,
        help="Folder of the pretrained weights of high magnification, FOLDER < `simclr/runs/[FOLDER]`",
    )
    parser.add_argument(
        "--weights_low",
        default=None,
        type=str,
        help="Folder of the pretrained weights of low magnification, FOLDER <`simclr/runs/[FOLDER]`",
    )
    parser.add_argument(
        "--tree_fusion",
        default="cat",
        type=str,
        help="Fusion method for high and low mag features in a tree method [cat|fusion]",
    )
    parser.add_argument(
        "--dataset",
        default="TCGA-lung-single",
        type=str,
        help="Dataset folder name [TCGA-lung-single]",
    )
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_transform = Compose([ToTensor()])
    i_classifier = None

    if args.backbone in ["resnet18", "resnet34", "resnet50", "resnet101"]:
        if args.norm_layer == "instance":
            norm = nn.InstanceNorm2d
            pretrain = False
        elif args.norm_layer == "batch":
            norm = nn.BatchNorm2d
            pretrain = args.weights == "ImageNet"
        else:
            raise ValueError(f"Unknown normalization layer: {args.norm_layer}")

        if args.backbone == "resnet18":
            resnet = models.resnet18(pretrained=pretrain, norm_layer=norm)
            num_feats = 512
        elif args.backbone == "resnet34":
            resnet = models.resnet34(pretrained=pretrain, norm_layer=norm)
            num_feats = 512
        elif args.backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrain, norm_layer=norm)
            num_feats = 2048
        elif args.backbone == "resnet101":
            resnet = models.resnet101(pretrained=pretrain, norm_layer=norm)
            num_feats = 2048

        for param in resnet.parameters():
            param.requires_grad = False
        resnet.fc = nn.Identity()

        if (
            args.magnification == "tree"
            and args.weights_high is not None
            and args.weights_low is not None
        ):
            i_classifier_h = mil.IClassifier(
                resnet, num_feats, output_class=args.num_classes
            ).to(device)
            i_classifier_l = mil.IClassifier(
                copy.deepcopy(resnet), num_feats, output_class=args.num_classes
            ).to(device)

            if (
                args.weights_high == "ImageNet"
                or args.weights_low == "ImageNet"
                or args.weights == "ImageNet"
            ):
                if args.norm_layer == "batch":
                    print("Use ImageNet features.")
                else:
                    raise ValueError(
                        "Please use batch normalization for ImageNet feature"
                    )
            else:
                weight_path = os.path.join(
                    "simclr", "runs", args.weights_high, "checkpoints", "model.pth"
                )
                state_dict_weights = torch.load(weight_path)
                for i in range(4):
                    state_dict_weights.popitem()
                state_dict_init = i_classifier_h.state_dict()
                new_state_dict = OrderedDict()
                for (k, v), (k_0, v_0) in zip(
                    state_dict_weights.items(), state_dict_init.items()
                ):
                    name = k_0
                    new_state_dict[name] = v
                i_classifier_h.load_state_dict(new_state_dict, strict=False)
                os.makedirs(os.path.join("embedder", args.dataset), exist_ok=True)
                torch.save(
                    new_state_dict,
                    os.path.join("embedder", args.dataset, "embedder-high.pth"),
                )

                weight_path = os.path.join(
                    "simclr", "runs", args.weights_low, "checkpoints", "model.pth"
                )
                state_dict_weights = torch.load(weight_path)
                for i in range(4):
                    state_dict_weights.popitem()
                state_dict_init = i_classifier_l.state_dict()
                new_state_dict = OrderedDict()
                for (k, v), (k_0, v_0) in zip(
                    state_dict_weights.items(), state_dict_init.items()
                ):
                    name = k_0
                    new_state_dict[name] = v
                i_classifier_l.load_state_dict(new_state_dict, strict=False)
                os.makedirs(os.path.join("embedder", args.dataset), exist_ok=True)
                torch.save(
                    new_state_dict,
                    os.path.join("embedder", args.dataset, "embedder-low.pth"),
                )
                print("Use pretrained features.")

        elif args.magnification in ["single", "high", "low"]:
            i_classifier = mil.IClassifier(
                resnet, num_feats, output_class=args.num_classes
            ).to(device)

            if args.weights == "ImageNet":
                if args.norm_layer == "batch":
                    print("Use ImageNet features.")
                else:
                    print("Please use batch normalization for ImageNet feature")
            else:
                if args.weights is not None:
                    weight_path = os.path.join(
                        "simclr", "runs", args.weights, "checkpoints", "model.pth"
                    )
                else:
                    weight_path = glob.glob("simclr/runs/*/checkpoints/*.pth")[-1]
                state_dict_weights = torch.load(weight_path)
                for i in range(4):
                    state_dict_weights.popitem()
                state_dict_init = i_classifier.state_dict()
                new_state_dict = OrderedDict()
                for (k, v), (k_0, v_0) in zip(
                    state_dict_weights.items(), state_dict_init.items()
                ):
                    name = k_0
                    new_state_dict[name] = v
                i_classifier.load_state_dict(new_state_dict, strict=False)
                os.makedirs(os.path.join("embedder", args.dataset), exist_ok=True)
                torch.save(
                    new_state_dict,
                    os.path.join("embedder", args.dataset, "embedder.pth"),
                )
                print("Use pretrained features.")
        else:
            raise ValueError(f"Unsupported magnification: {args.magnification}")

    elif args.backbone == "h-optimus-1":
        i_classifier, feature_transform = load_hoptimus1_feature_extractor()
        num_feats = 1536
        if args.weights is not None:
            print(
                "Warning: --weights is ignored for the h-optimus-1 feature extractor backbone."
            )

    elif args.backbone == "UNI2-h":
        i_classifier, feature_transform = load_uni2h_feature_extractor()
        num_feats = 1536
        if args.weights is not None:
            print(
                "Warning: --weights is ignored for the UNI2-h feature extractor backbone."
            )

    elif args.backbone == "Virchow2":
        i_classifier, feature_transform = load_virchow2_feature_extractor()
        num_feats = 2560
        if args.weights is not None:
            print(
                "Warning: --weights is ignored for the Virchow2 feature extractor backbone."
            )

    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    if (
        args.backbone in ["h-optimus-1", "UNI2-h", "Virchow2"]
        and args.magnification == "tree"
    ):
        raise NotImplementedError(
            "Tree magnification is not supported for timm feature extractor backbones."
        )

    if (
        args.magnification == "tree"
        or args.magnification == "low"
        or args.magnification == "high"
    ):
        bags_path = os.path.join("WSI", args.dataset, "pyramid", "*", "*")
    else:
        bags_path = os.path.join("WSI", args.dataset, "single", "*", "*")
    feats_path = os.path.join("datasets", args.dataset)

    os.makedirs(feats_path, exist_ok=True)
    bags_list = glob.glob(bags_path)

    if args.magnification == "tree":
        compute_tree_feats(args, bags_list, i_classifier_l, i_classifier_h, feats_path)
    else:
        compute_feats(
            args,
            bags_list,
            i_classifier,
            feats_path,
            args.magnification,
            transform=feature_transform,
        )
    n_classes = glob.glob(os.path.join("datasets", args.dataset, "*" + os.path.sep))
    n_classes = sorted(n_classes)
    all_df = []
    for i, item in enumerate(n_classes):
        bag_csvs = glob.glob(os.path.join(item, "*.csv"))
        bag_df = pd.DataFrame(bag_csvs)
        bag_df["label"] = i
        bag_df.to_csv(
            os.path.join("datasets", args.dataset, item.split(os.path.sep)[2] + ".csv"),
            index=False,
        )
        print(
            f"Saved {len(bag_csvs)} bags for class {i} in {item.split(os.path.sep)[2]}.csv"
        )
        all_df.append(bag_df)
    bags_path = pd.concat(all_df, axis=0, ignore_index=True)
    bags_path = shuffle(bags_path)
    bags_path.to_csv(
        os.path.join("datasets", args.dataset, args.dataset + ".csv"), index=False
    )


if __name__ == "__main__":
    main()
