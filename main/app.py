import librosa

filename = librosa.example('nutcracker')

y, sr = librosa.load(filename)

print(y)

