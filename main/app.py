import librosa
import numpy as np
from pathlib import Path
import pandas as pd

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}

def stats_named(feature: np.ndarray, name: str):
    """Collapse a (n_rows, n_frames) feature into named mean/std values."""
    means = feature.mean(axis=1)
    stds = feature.std(axis=1)
    names = [f"{name}_{i}_mean" for i in range(len(means))] + \
            [f"{name}_{i}_std" for i in range(len(stds))]
    values = np.concatenate([means, stds])
    return names, values

def extract_features(song: Path):
    y, sr = librosa.load(song, sr=None)
    y_harmonic = librosa.effects.harmonic(y)
 
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    rms = librosa.feature.rms(y=y)
 
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
 
    names, values = [], []
    for feature, label in [
        (mfcc, "mfcc"),
        (chroma, "chroma"),
        (contrast, "contrast"),
        (tonnetz, "tonnetz"),
        (centroid, "centroid"),
        (rolloff, "rolloff"),
        (zcr, "zcr"),
        (rms, "rms"),
    ]:
        n, v = stats_named(feature, label)
        names.extend(n)
        values.extend(v)
 
    names.append("tempo")
    values.append(tempo)
 
    return names, np.array(values)


def extract_features_from_folder(songs_folder: Path, output_path: Path = Path("features_raw.pkl")):
    song_files = [
        p for p in songs_folder.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    total = len(song_files)
 
    # Resume from a previous run instead of re-extracting everything
    records = []
    already_processed = set()
    if output_path.exists():
        existing_df = pd.read_pickle(output_path)
        already_processed = set(existing_df["path"])
        records = existing_df.to_dict("records")
        print(f"Resuming: {len(already_processed)} song(s) already processed.")
 
    failed = []
    for i, song in enumerate(song_files, start=1):
        song_path = str(song.resolve())
        if song_path in already_processed:
            continue
 
        print(f"[{i}/{total}] extracting: {song.name}")
        try:
            names, values = extract_features(song)
        except Exception as e:
            print(f"  failed: {e}")
            failed.append(song_path)
            continue
 
        record = {"path": song_path}
        record.update(dict(zip(names, values)))
        records.append(record)
 
    df = pd.DataFrame(records)
    df.to_pickle(output_path)
 
    if failed:
        print(f"\n{len(failed)} song(s) failed to process:")
        for f in failed:
            print(f"  {f}")
 
    return df

if __name__ == "__main__":
    song_files = Path("../song_files")
    df = extract_features_from_folder(song_files)
    print(f"\nProcessed {df.shape[0]} songs, {df.shape[1]} columns.")
    

    for row in df.iterrows():
        print(df.iloc[0]['path'])

