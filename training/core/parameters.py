import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6

CROSS_ENTROPY_CLASS_WEIGHTS = torch.tensor([1.0, 174.9, 43.8, 111.0], dtype=torch.float32)

BATCH_SIZE = 4
NUM_WORKERS = 8
LOG_EVERY_BATCHES = 200  # file logging cadence (not noisy)
ACCUM_STEPS = 2          # gradient accumulation → effective batch = BATCH_SIZE × ACCUM_STEPS

SEED = 1337
