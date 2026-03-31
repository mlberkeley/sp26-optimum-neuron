
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_REPO = "Wan-AI/Wan2.2-TI2V-5B"


def main():
    script_dir = Path(__file__).resolve().parent
    wan_root = script_dir.parent
    ckpt_root = wan_root / "ckpts"
    local_dir = ckpt_root / "Wan2.2-TI2V-5B"

    ckpt_root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {MODEL_REPO}")
    print(f"Destination: {local_dir}")

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print("Done.")
    print(f"Checkpoint dir: {local_dir}")


if __name__ == "__main__":
    main()