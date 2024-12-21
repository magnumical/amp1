import numpy as np
import librosa

def add_noise(data, x):
    """Add noise to audio data."""
    noise = np.random.randn(len(data))
    return data + x * noise

def shift(data, x):
    """Shift audio data."""
    return np.roll(data, int(x))

def stretch(data, rate):
    """Stretch audio data."""
    return librosa.effects.time_stretch(data, rate=rate)

def pitch_shift(data, rate):
    """Apply pitch shifting to audio data."""
    return librosa.effects.pitch_shift(data, sr=22050, n_steps=rate)
