from pathlib import Path

# --- Paths ---

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "library.db"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"

DATA_DIR.mkdir(exist_ok=True)

# --- Audio ---

SAMPLE_RATE = 22_050
MONO = True
ANALYSIS_DURATION = 60  # seconds; set to None to analyse full track

# --- Feature extraction ---

N_MFCC = 13
N_CHROMA = 12
HOP_LENGTH = 512
N_FFT = 2048