import os
from huggingface_hub import login, hf_hub_download

login()

models = {
    "uni2h": "MahmoodLab/UNI2-h",
    "virchow2": "paige-ai/Virchow2",
    "hoptimus1": "bioptimus/H-optimus-1",
}

for name, repo in models.items():

    local_dir = f"./models/{name}"
    os.makedirs(local_dir, exist_ok=True)

    try:
        hf_hub_download(
            repo_id=repo,
            filename="pytorch_model.bin",
            local_dir=local_dir,
            force_download=True,
        )
    except:
        pass

    try:
        hf_hub_download(
            repo_id=repo,
            filename="model.safetensors",
            local_dir=local_dir,
            force_download=True,
        )
    except:
        pass

    try:
        hf_hub_download(
            repo_id=repo,
            filename="config.json",
            local_dir=local_dir,
            force_download=True,
        )
    except:
        pass

print("All model files downloaded.")
