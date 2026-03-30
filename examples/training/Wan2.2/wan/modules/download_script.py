from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Wan-AI/Wan2.2-TI2V-5B",
    filename="Wan2.2_VAE.pth",
    local_dir="./Wan2.2-TI2V-5B"
)

print("downloaded to:", path)