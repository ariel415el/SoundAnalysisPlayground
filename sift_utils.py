from math import ceil

import numpy as np


def hz2mel(freq):
	return (2595 * np.log10(1 + freq / 700))


def mel2hz(mel):
	return (700 * (10 ** (mel / 2595) - 1))


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


def pre_emphasize(signal, amplitude=0.97):
	"""
	Pre ephasize increases the amplitude high frequencies and and decreases it for low frequencies.
	This adds to the balance of the signal as low frequncies usually have smaller magnitudes.
	"""
	return np.append(signal[0], signal[1:] - amplitude * signal[:-1])