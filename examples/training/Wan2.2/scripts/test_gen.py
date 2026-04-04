
import subprocess
import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent
    wan_root = script_dir.parent
    ckpt_dir = wan_root / "ckpts" / "Wan2.2-TI2V-5B"
    out_dir = wan_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = wan_root / "examples" / "i2v_input.JPG"
    output_path = out_dir / "ti2v_smoke.mp4"

    if not (wan_root / "generate.py").exists():
        raise FileNotFoundError(f"Could not find generate.py in {wan_root}")

    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}\n"
            f"Run download_ti2v_weights.py first."
        )

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    prompt = (
        "A cinematic medium shot of a lighthouse on a rocky coast during sunset, "
        "waves crashing, realistic motion, filmic detail."
    )

    cmd = [
        sys.executable,
        "generate.py",
        "--task", "ti2v-5B",
        "--size", "1280*704",
        "--ckpt_dir", str(ckpt_dir),
        "--offload_model", "True",
        "--convert_model_dtype",
        "--t5_cpu",
        "--image", str(image_path),
        "--prompt", prompt,
        "--base_seed", "1234",
        "--frame_num", "17",
        "--sample_steps", "8",
        "--save_file", str(output_path),
    ]

    print("Running command:")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    print()

    subprocess.run(cmd, cwd=wan_root, check=True)

    print()
    print(f"Done. Output saved to: {output_path}")


if __name__ == "__main__":
    main()