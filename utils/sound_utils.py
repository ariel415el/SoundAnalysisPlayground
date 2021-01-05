import librosa
import numpy as np


def hz2mel(freq):
    return (2595 * np.log10(1 + freq / 700))


def mel2hz(mel):
    return (700 * (10 ** (mel / 2595) - 1))


def get_mel_filter_bank(sample_rate, num_filters, nfft, low_freq_mel_in_hz=0, high_freq_mel_in_hz=None):
    """
    Creates a set of frequency filters in the mel scale
    :param sample_rate: the sample rate of the data
    :param nfft:
    :return: numpy array of shape num_filters x nfft / 22
    """

    if high_freq_mel_in_hz is None:
        high_freq_mel_in_hz = sample_rate / 2

    mel_bins = np.linspace(hz2mel(low_freq_mel_in_hz), hz2mel(high_freq_mel_in_hz), num_filters + 2)  # Equally spaced in Mel scale
    hz_bins = mel2hz(mel_bins)
    bin_indices = np.floor((nfft + 1) * hz_bins / sample_rate)

    fbank = np.zeros((num_filters, int(np.floor(nfft / 2 + 1))))
    for m in range(1, num_filters + 1):
        f_m_minus = int(bin_indices[m - 1])  # left
        f_m = int(bin_indices[m])  # center
        f_m_plus = int(bin_indices[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_indices[m - 1]) / (bin_indices[m] - bin_indices[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_indices[m + 1] - k) / (bin_indices[m + 1] - bin_indices[m])

    hz_bin_centers = hz_bins[1:-1]
    return fbank, hz_bin_centers


def mel_filter(power_spectogram, nfft, n_mel_bins, mel_min_freq, mel_max_freq, sample_rate, use_librosa=True):
    if use_librosa:
        mel_filter_bank = librosa.filters.mel(
            sr=sample_rate,
            n_fft=nfft,
            n_mels=n_mel_bins,
            fmin=mel_min_freq,
            fmax=mel_max_freq)

    else:
        mel_filter_bank, _ = get_mel_filter_bank(sample_rate, num_filters=n_mel_bins, nfft=nfft,
                                                 low_freq_mel_in_hz=mel_min_freq, high_freq_mel_in_hz=mel_max_freq)

    mel_spectrogram = np.dot(mel_filter_bank, power_spectogram)

    return mel_spectrogram


def to_decibles(spectogram, use_librosa):
    if use_librosa:
        decible_spectogram = librosa.core.power_to_db(spectogram, ref=1.0, amin=1e-10, top_db=None).astype(np.float32)
    else:
        mel_spectrogram = np.where(spectogram == 0, np.finfo(float).eps, spectogram)  # Numerical Stability
        decible_spectogram = 20 * np.log10(mel_spectrogram)  # dB
    return decible_spectogram