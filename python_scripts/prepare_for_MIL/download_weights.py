import os
from huggingface_hub import login, hf_hub_download

login()

local_dir = "./uni2h_model"
os.makedirs(local_dir, exist_ok=True)

hf_hub_download(
    repo_id="MahmoodLab/UNI2-h",
    filename="pytorch_model.bin",
    local_dir=local_dir,
    force_download=True,
)

hf_hub_download(
    repo_id="MahmoodLab/UNI2-h",
    filename="config.json",
    local_dir=local_dir,
    force_download=True,
)

print("Download complete.")
