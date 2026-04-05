# Speech Emotion Recognition

Classify human speech into 7 emotional categories using deep-learning sequence models trained on the **TESS** (Toronto Emotional Speech Set) dataset.

Four architectures are implemented and compared:

| Model | Description |
|-------|-------------|
| LSTM | Unidirectional Long Short-Term Memory |
| BiLSTM | Bidirectional LSTM |
| GRU | Gated Recurrent Unit |
| BiGRU | Bidirectional GRU |

---

## Dataset

**TESS** — 2,800 recordings across 7 emotions (400 per class):
`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

Download from Kaggle:
```
kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess
```

Set the environment variable before running:
```bash
export SER_DATA_ROOT=/path/to/speech-emotion-recognition-en
```

---

## Project Structure

```
Speech_Emotion_Recognition/
├── config.py              # All hyperparameters and paths
├── main.py                # Entry point — data → features → train → evaluate
├── requirements.txt
├── src/
│   ├── data_loader.py     # Load TESS into a DataFrame
│   ├── augmentation.py    # Noise, time-stretch, pitch-shift
│   ├── features.py        # MFCC extraction + feature matrix builder
│   ├── visualization.py   # Waveforms, spectrograms, MFCC heatmap, training curves
│   ├── models.py          # Model factory (LSTM / BiLSTM / GRU / BiGRU)
│   └── train.py           # Training loop + evaluation + model saving
└── outputs/
    ├── models/            # Saved .keras model files
    └── plots/             # All generated figures
```

---

## Setup

```bash
git clone https://github.com/<your-username>/Speech_Emotion_Recognition.git
cd Speech_Emotion_Recognition
pip install -r requirements.txt
```

---

## Usage

**Train all four models:**
```bash
python main.py
```

**Train specific models only:**
```bash
python main.py --models lstm bilstm
```

**Skip exploratory visualizations (faster iteration):**
```bash
python main.py --skip-viz
```

---

## Pipeline

```
TESS Audio Files
      │
      ▼
Data Loading (data_loader.py)
  └── DataFrame: [Emotions, Path]
      │
      ▼
Feature Extraction (features.py)
  └── 40-dim MFCC per clip  →  X: (2800, 40, 1)
  └── One-hot labels         →  y: (2800, 7)
      │
      ▼
Model Training (models.py + train.py)
  └── LSTM / BiLSTM / GRU / BiGRU
  └── 300 epochs, batch 64, 80/20 train-val split
      │
      ▼
Evaluation + Outputs
  └── Classification report (per-class precision / recall / F1)
  └── Confusion matrix heatmap
  └── Training accuracy & loss curves
  └── Saved .keras model files
```

---

## Feature Engineering

**MFCC (Mel-Frequency Cepstral Coefficients):**  
Each audio clip is loaded at 22 050 Hz, trimmed to 3 seconds with a 0.5 s offset, and summarised as the time-averaged 40-dimensional MFCC vector — a compact representation of the spectral envelope of speech.

**Data augmentation** (visualised in exploratory plots):
- Gaussian noise injection
- Time-stretching (0.8×)
- Pitch-shifting (±2 semitones)

---

## Model Architecture

All four models share the same head after the recurrent layer:

```
Recurrent layer (256 units)  →  Dropout(0.2)
Dense(128, ReLU)             →  Dropout(0.2)
Dense(64, ReLU)              →  Dropout(0.2)
Dense(7, Softmax)
```

---

## Results

Training is run for 300 epochs on the full TESS dataset. Outputs (plots + classification reports) are written to `outputs/`.

---

## Tech Stack

- **Python 3.10+**
- TensorFlow / Keras
- librosa
- scikit-learn
- pandas, NumPy, Matplotlib, Seaborn
