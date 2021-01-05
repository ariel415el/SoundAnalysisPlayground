from math import ceil

import librosa
import numpy as np



def split_to_frames(signal, frame_length, overlap_length):
    """
    Split the signal into overlapping frames. pads the signal if necessary for division.
    Here a new frame of size "frame_length" samples starts every "overlap_length" samples
    :param frame_length: how many samples in each frames
    :param overlap_length: overlapping samples
    :return: numpy array of size num_frames, frame_length
    """
    num_frames = ceil(len(signal) / float(overlap_length))
    pad_size = (num_frames - 1) * overlap_length + frame_length - len(signal)
    padded_signal = np.append(signal, np.zeros(pad_size))

    # extract overlapping frames:
    frame_offsets = np.tile(np.arange(0, num_frames) * overlap_length, (frame_length, 1)).T
    frame_indices = np.tile(np.arange(0, frame_length), (num_frames, 1))
    frame_indices += frame_offsets

    return padded_signal[frame_indices]


def pre_emphasize(signal, amplitude=0.97):
    """
    Pre ephasize increases the amplitude high frequencies and and decreases it for low frequencies.
    This adds to the balance of the signal as low frequncies usually have smaller magnitudes.
    """
    return np.append(signal[0], signal[1:] - amplitude * signal[:-1])


def complex_to_power_spectogram(complex_spectogram, nfft):
    magnitude_spectogram = np.absolute(complex_spectogram)
    power_spectrum = (1.0 / nfft) * (magnitude_spectogram ** 2)  # Power Spectrum: Normalizes the magnitudes by the frequency resolution
    return power_spectrum


def sift(samples, window_size, hop_size, nfft, use_librosa=True):
    if use_librosa:
        complex_spectogran =  librosa.core.stft(
            y=samples,
            n_fft=nfft,
            hop_length=hop_size,
            win_length=window_size,
            window=np.hanning(window_size),
            center=True,
            dtype=np.complex64,
            pad_mode='reflect')
    else:
        frames = split_to_frames(samples, window_size, hop_size)  # shape=(num_frames, window_size)

        # Windowing: apply apply a filter to force continous signal in each frame by reducing the frame extremes to zero
        frames *= np.hanning(window_size)

        complex_spectogran = np.fft.rfft(frames, nfft).T

    return complex_spectogran  # shape=(nfft//2+1, num_frames)