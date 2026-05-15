from torchmil.datasets import ProcessedMILDataset

features_path = "WSI/IgA/UNI2-h_feats"
labels_path = "WSI/IgA/labels"
coords_path = "WSI/IgA/coords"

val_features_path = "WSI/IgA/UNI2-h_feats_val"
val_labels_path = "WSI/IgA/labels_val"
val_coords_path = "WSI/IgA/coords_val"

dataset = ProcessedMILDataset(
    features_path=features_path,
    labels_path=labels_path,
    coords_path=coords_path,
    bag_keys=["X", "Y", "adj", "coords"],
)
print(f"Number of bags training set: {len(dataset)}")


val_dataset = ProcessedMILDataset(
    features_path=val_features_path,
    labels_path=val_labels_path,
    coords_path=val_coords_path,
    bag_keys=["X", "Y", "adj", "coords"],
)
print(f"Number of bags validation set: {len(val_dataset)}")

bag = dataset[0]
for key in bag.keys():
    print(f"{key}: {bag[key].shape}")

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torchmil.data import collate_fn

train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=128, shuffle=True, collate_fn=collate_fn
)
test_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn
)

from torchmil.models import abmil

embedding_dim = 1536  # feature extractor specific

model = ABMIL(emb_dim=embedding_dim, att_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")


def train(dataloader, epoch):
    model.train()

    sum_loss = 0.0
    sum_correct = 0.0
    for batch in dataloader:
        batch = batch.to(device)
        out = model(batch["X"], batch["mask"])
        loss = criterion(out, batch["Y"].float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        sum_loss += loss.item()
        pred = (out &gt; 0).float()
        sum_correct += (pred == batch["Y"]).sum().item()
        sum_loss += loss.item()

    print(
        f"[Epoch {epoch}] Train, train/loss: {sum_loss / len(dataloader)}, 'train/bag/acc': {sum_correct / len(dataloader.dataset)}"
    )


def val(dataloader, epoch):
    model.eval()

    sum_loss = 0.0
    sum_correct = 0.0
    for batch in dataloader:
        batch = batch.to(device)
        out = model(batch["X"], batch["mask"])
        loss = criterion(out, batch["Y"].float())

        sum_loss += loss.item()
        pred = (out &gt; 0).float()
        sum_correct += (pred == batch["Y"]).sum().item()
        sum_loss += loss.item()

    print(
        f"[Epoch {epoch}] Validation, val/loss: {sum_loss / len(dataloader)}, 'val/bag/acc': {sum_correct / len(dataloader.dataset)}"
    )


model = model.to(device)
for epoch in range(20):
    train(train_dataloader, epoch + 1)
    val(test_dataloader, epoch + 1)