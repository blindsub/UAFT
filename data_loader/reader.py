import librosa

def read_wav(fname, return_rate=False):
    src, sr = librosa.load(fname, sr=8000)
    if return_rate:
        return src.squeeze(), sr
    else:
        return src.squeeze()

def write_wav(fname, src, sample_rate):
	librosa.output.write_wav(fname, src, sample_rate)