###################################################################################################
# Code inspired by https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
###################################################################################################
import numpy as np
from math import ceil
import os

import soundfile

from utils.fft_utils import split_to_frames, pre_emphasize
from utils.plot_utils import plot_filters, plot_signals, plot_filters_reaction
from utils.sound_utils import get_mel_filter_bank


FRAME_SIZE = 512
FRAME_STRIDE = 256
NFFT = 512
NUM_MEL_FILTES=40

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_multi_signal(amplitudes, frequencies, phases, time, sampling_rate=8000, noise_factor=0.0):
	"""
	Additive construction of a waveform from specified harmonic signals
	"""
	signal_name = ""
	xs = np.arange(0, time, 1/sampling_rate)
	samples = []
	for i, (a, f, p) in enumerate(zip(amplitudes, frequencies, phases)):
		signal = a * np.sin(2 * np.pi * f * (xs + p))
		samples.append(signal)
		if i > 0:
			signal_name += "+"
		signal_name += f"({a},{f},{p})"
	signal = np.mean(samples, axis=0)

	signal += noise_factor*np.random.normal(0,1,len(signal))

	return xs, signal, sampling_rate, signal_name


def load_signal_from_file(sound_file):
	signal, sample_rate = soundfile.read(sound_file)  # File assumed to be in the same directory
	if len(signal.shape) > 1 and signal.shape[1] > 1:
		print("Using only first signal out of more")
		signal = signal[:, 0] # use only one signal if there are more than one
	ts = np.arange(0, len(signal) / sample_rate, 1 / sample_rate)

	return ts, signal, sample_rate


def analyze_synthetic_signals():
	"""
	Adds various harmonic waves into a fabricate a wave form and shows the spectogram that indicates the componnens
	"""
	signal_xs, signal_ys, sample_rate, signal_name = create_multi_signal([1, 1, 1], [10, 20, 40], [0, 0, 0], time=5,
															sampling_rate=100, noise_factor=0.5)
	# signal_xs, signal_ys, sample_rate = create_multi_signal([1], [7], [0], sampling_rate=100)

	maximal_frequency = ceil(sample_rate/2) # maximal detected frequency (Nyquist frequency)

	ft = np.absolute(np.fft.fft(signal_ys))
	# Since our signal is Real and not Complex it is symetric around the 0 frequency and we
	# can take only half of it or rather use np.fft.rfft
	num_frequencies = ceil(len(signal_ys)/2)
	frequencies = np.linspace(0, maximal_frequency, num_frequencies)
	ft = ft[:num_frequencies]
	plot_signals([signal_xs, frequencies], [signal_ys, ft], [signal_name, "ft"], "synthetic_wave-form.png")


def plot_signal_spectogram(sound_file):
	"""
	Plos the spectorgram of a given sound file
	"""
	signal_xs, signal_ys, sample_rate = load_signal_from_file(sound_file)

	signal_ys_emph = pre_emphasize(signal_ys)

	# Plot signal
	plot_signals([signal_xs, signal_xs], [signal_ys, signal_ys_emph], ["origninal signal", "emphasised signal"], "original_+emph_waveform.png")

	frames = split_to_frames(signal_ys_emph, int(FRAME_SIZE * sample_rate), int(FRAME_STRIDE * sample_rate))

	# Windowing: apply apply a filter to force continous signal in each frame by reducing the frame extremes to zero
	frames *= np.hamming(int(FRAME_SIZE * sample_rate))

	mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
	pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum: Normalizes the magnitudes by the frequency resolution

	# plot concatenated frames (with repetition)
	plot_signals([range(len(frames.reshape(-1))), range(len(mag_frames.reshape(-1))), range(len(mag_frames.reshape(-1)))],
				 [frames.reshape(-1), mag_frames.reshape(-1), pow_frames.reshape(-1)],
				 ["concatenated frames)", "concatenated mag_frames", "concatenated power_frames"],"org_mag_pos_concatenated_frames.png")


	mel_filter_bank, hz_bin_centers = get_mel_filter_bank(sample_rate, num_filters=NUM_MEL_FILTES, nfft=NFFT)

	# Plot Mel filters
	plot_filters(mel_filter_bank, "Mel-filters.png")

	filter_outputs = np.dot(pow_frames, mel_filter_bank.T)
	filter_outputs = np.where(filter_outputs == 0, np.finfo(float).eps, filter_outputs)  # Numerical Stability
	filter_outputs = 20 * np.log10(filter_outputs)  # dB

	plot_filters_reaction(filter_outputs, hz_bin_centers, "Spectogram.png")

if __name__ == '__main__':
	analyze_synthetic_signals()
	# plot_signal_spectogram('OSR_us_000_0010_8k.wav')
	# plot_signal_spectogram('sound_snippets/1A-T001_clap_cropped.WAV')
	# plot_signal_spectogram('sound_snippets/2B-T003_doorslam.WAV')