from pathlib import Path

# ---------- paths ----------
ROOT = Path(__file__).resolve().parent
WAN_ROOT = ROOT.parent
CKPT_DIR = WAN_ROOT / "ckpts" / "Wan2.2-TI2V-5B"
IMAGE_PATH = WAN_ROOT / "examples" / "i2v_input.JPG"
OUTPUT_DIR = WAN_ROOT / "outputs"

# ---------- device/runtime ----------
DEVICE = "cuda"  # SET DEVICE!!!!
DEVICE_ID = 0
RANK = 0

OFFLOAD_MODEL = True
T5_CPU = True
CONVERT_MODEL_DTYPE = True
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

# frame number should be 4n + 1 so (frame_number - 1) should be a multiple of 4 (1 anchor frame)
# reason: one latent time step corresponds to 4 output frames and +1 for anchor/initial frame

# bigger numbers
# FRAME_NUM = 81
# SAMPLE_STEPS = 24

# smaller numbers
FRAME_NUM = 17
SAMPLE_STEPS = 8

SAMPLE_SOLVER = "unipc"
BASE_SEED = 1234

# set to None to use Wan defaults from config
SAMPLE_SHIFT = None
SAMPLE_GUIDE_SCALE = None

# ---------- profiling ----------
ENABLE_PROFILING = True
PROFILE_OUTPUT_DIR = OUTPUT_DIR / "profiles"
PROFILE_MIN_TIME_MS = 0.0  # don't output profiled nodes with total_time (s) < PROFILE_MIN_TIME_MS / 1000.0
