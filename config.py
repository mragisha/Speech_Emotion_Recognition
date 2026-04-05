import os

# ── Dataset ────────────────────────────────────────────────────────────────────
ROOT_DIR = os.getenv("SER_DATA_ROOT", "/kaggle/input/speech-emotion-recognition-en")
TESS_PATH = os.path.join(ROOT_DIR, "Tess")

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
TARGET_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ── Feature extraction ─────────────────────────────────────────────────────────
SAMPLE_DURATION = 3       # seconds loaded per clip
SAMPLE_OFFSET   = 0.5     # seconds skipped at start
N_MFCC          = 40      # number of MFCC coefficients

# ── Model ──────────────────────────────────────────────────────────────────────
LSTM_UNITS   = 256
DENSE_UNITS  = [128, 64]
DROPOUT_RATE = 0.2

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS           = 300
BATCH_SIZE       = 64
VALIDATION_SPLIT = 0.2
RANDOM_SEED      = 42

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = "outputs"
MODEL_SAVE   = os.path.join(OUTPUT_DIR, "models")
PLOT_SAVE    = os.path.join(OUTPUT_DIR, "plots")
CSV_SAVE     = os.path.join(OUTPUT_DIR, "Tess_df.csv")
