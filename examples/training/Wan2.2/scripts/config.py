# config.py
from pathlib import Path

# ---------- paths ----------
ROOT = Path(__file__).resolve().parent
WAN_ROOT = ROOT.parent
CKPT_DIR = WAN_ROOT / "ckpts" / "Wan2.2-TI2V-5B"
IMAGE_PATH = WAN_ROOT / "examples" / "i2v_input.JPG"
OUTPUT_DIR = WAN_ROOT / "outputs"

# ---------- device/runtime ----------
# Examples: "cuda", "cpu", "neuron"
DEVICE = "cpu"

# Only used if DEVICE is an indexed device type like CUDA / neuronx-style local rank flows.
DEVICE_ID = 0
RANK = 0

OFFLOAD_MODEL = True
T5_CPU = True
CONVERT_MODEL_DTYPE = True

# distributed/model flags
T5_FSDP = False
DIT_FSDP = False
USE_SP = False

# ---------- run mode ----------
SKIP_SAVE = False

# ---------- generation ----------
PROMPT = (
    "A cinematic medium shot of a lighthouse on a rocky coast during sunset, "
    "waves crashing, realistic motion, filmic detail."
)

SIZE = "1280*704"
FRAME_NUM = 17
SAMPLE_STEPS = 8
SAMPLE_SOLVER = "unipc"   # "unipc" or "dpm++"
BASE_SEED = 1234

# set to None to use Wan defaults from config
SAMPLE_SHIFT = None
SAMPLE_GUIDE_SCALE = None