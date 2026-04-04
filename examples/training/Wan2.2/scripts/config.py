from pathlib import Path

# ---------- paths ----------
ROOT = Path(__file__).resolve().parent
WAN_ROOT = ROOT.parent
CKPT_DIR = WAN_ROOT / "ckpts" / "Wan2.2-TI2V-5B"
IMAGE_PATH = WAN_ROOT / "examples" / "i2v_input.JPG"
OUTPUT_DIR = WAN_ROOT / "outputs"

# ---------- run mode ----------
PROFILE_WITH_NCU = False
SKIP_SAVE = False
WARMUP = False

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

# ---------- model/runtime ----------
OFFLOAD_MODEL = True
T5_CPU = True
CONVERT_MODEL_DTYPE = True

# ---------- optional metrics ----------
PEAK_TFLOPS = None  # set if you want MFU
NVTX_RANGE = "wan_ti2v_generate"
