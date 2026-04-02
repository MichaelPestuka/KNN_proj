# Number of past frames/actions we provide
BUFFER_SIZE = 9

HEIGHT = 240
WIDTH = 320

# CFG ratio
CFG_GUIDANCE_SCALE = 1.5

# Default number of inference steps for diffusion
DEFAULT_NUM_INFERENCE_STEPS = 10

# There are 10 noise buckets total (inlined from utils.py for inference-only)
NUM_BUCKETS = 10

PRETRAINED_MODEL_NAME_OR_PATH = "CompVis/stable-diffusion-v1-4"
