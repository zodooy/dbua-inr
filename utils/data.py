from pathlib import Path
from hdf5storage import loadmat
from torch.utils import data

MAKE_VIDEO = False
MAKE_FIGURE = True
OPTIMIZE_GLOBAL_SOS = False

Z_GROWING = False
RANDOM_PATCHING = True

N_ITERS = 300
LEARNING_RATE = 0.5
ASSUMED_C = 1540  # [m/s]

NPTS_IMAGE = 64
NPTS_PATCH = 64

# B-mode limits in m
BMODE_X_MIN = -12e-3
BMODE_X_MAX = 12e-3
BMODE_Z_MIN = 0e-3
BMODE_Z_MAX = 40e-3

# Sound speed grid in m
SOUND_SPEED_X_MIN = -16e-3
SOUND_SPEED_X_MAX = 16e-3
SOUND_SPEED_Z_MIN = 0e-3
SOUND_SPEED_Z_MAX = 44e-3
SOUND_SPEED_NXC = 19
SOUND_SPEED_NZC = 31
SOUND_SPEED_MAX = 1700
SOUND_SPEED_MIN = 1400

LAMBDA_TV = 1e2

NXK, NZK = 5, 5 # Phase estimate kernel size in samples
NXP, NZP = 15, 15 # Phase estimate patch grid size in samples
N_PATCHES = NXP * NZP

PHASE_ERROR_X_MIN = -20e-3
PHASE_ERROR_X_MAX = 20e-3
PHASE_ERROR_Z_MIN = 4e-3
PHASE_ERROR_Z_MAX = 44e-3

# PHASE_ERROR_X_MIN = -20e-3
# PHASE_ERROR_X_MAX = 20e-3
# PHASE_ERROR_Z_MIN = 4e-3
# PHASE_ERROR_Z_MAX = 44e-3

# Loss options
# -"pe" for phase error
# -"sb" for speckle brightness
# -"cf" for coherence factor
# -"lc" for lag one coherence

LOSS = "pe"

# Data options:
# (Constant Phantoms)
# - 1420
# - 1465
# - 1480
# - 1510
# - 1540
# - 1555
# - 1570
# (Heterogeneous Phantoms)
# - inclusion
# - inclusion_layer
# - four_layer
# - two_layer
# - checker2
# - checker8

SAMPLE = "inclusion_layer"

CTRUE = {
    "1420": 1420,
    "1465": 1465,
    "1480": 1480,
    "1510": 1510,
    "1540": 1540,
    "1555": 1555,
    "1570": 1570,
    "inclusion": 0,
    "inclusion_layer": 0,
    "four_layer": 0,
    "two_layer": 0,
    "checker2": 0,
    "checker8": 0
}

# Refocused plane wave datasets from base dataset directory
DATA_DIR = Path("./data")

def normalize(c):
    return (c - SOUND_SPEED_MIN) / (SOUND_SPEED_MAX - SOUND_SPEED_MIN)

def denormalize(c):
    return c * (SOUND_SPEED_MAX - SOUND_SPEED_MIN) + SOUND_SPEED_MIN

def load_dataset(sample):
    mdict = loadmat(f"{DATA_DIR}/{sample}.mat")
    iqdata = mdict["iqdata"]
    fs = mdict["fs"][0, 0]  # Sampling frequency
    fd = mdict["fd"][0, 0]  # Demodulation frequency
    dsf = mdict["dsf"][0, 0]  # Downsampling factor
    t = mdict["t"]  # time vector
    t0 = mdict["t0"]  # time zero of transmit
    elpos = mdict["elpos"]  # element position
    return iqdata, t0, fs, fd, elpos, dsf, t

# Custom Dataset for coordinates
class CoordDataset(data.Dataset):
    def __init__(self, coords):
        self.coords = coords

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        return self.coords[idx]