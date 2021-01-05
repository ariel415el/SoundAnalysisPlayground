###################################################################################################
# Code inspired by https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
###################################################################################################
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
from math import ceil
import os

from sift_utils import split_to_frames, get_mel_filter_bank, pre_emphasize

COLORS=plt.get_cmap("jet")

FRAME_SIZE = 0.03
FRAME_STRIDE = 0.015

NFFT = 512
NUM_MEL_FILTES=40
SINGAL_TIME = None # in seconds None for whole signal

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_filters(filters, plot_path):
	fig = plt.figure(figsize=(20,2))
	for i,filter in enumerate(filters):
		actual_indices = np.where(filter != 0)[0]
		actual_indices = np.concatenate(([actual_indices.min() - 1], actual_indices, [actual_indices.max() + 1]))
		plt.plot(actual_indices, filter[actual_indices], color=COLORS(i /len(filters)))
	plt.legend(ncol=int(len(filters)/4), loc='lower center', bbox_to_anchor=(1.2,0.5))
	plt.savefig(os.path.join(OUTPUT_DIR,plot_path))
	plt.clf()


def plot_signals(xs, signals, names, plot_path):
	fig = plt.figure(figsize=(20,2))
	for i, (x, signal,name) in enumerate(zip(xs, signals, names)):
		ax = plt.subplot(len(signals), 1, i+1)
		ax.plot(x, signal, label=name, color=COLORS(i /len(signals)))
		ax.legend(ncol=int(len(signals)/4))
	plt.savefig(os.path.join(OUTPUT_DIR,plot_path))
	plt.clf()


def plot_filters_reaction(filters_output, hz_bin_centers, plot_path):
	"""
	:param filters_output: num_frames x num_filters nd_matrix
	"""
	fig = plt.figure(figsize=(10,2))
	filters_output = filters_output.T
	plt.imshow(filters_output, cmap=plt.cm.jet)

	plt.xlabel("frame number")
	yticks_indices = np.arange(0, filters_output.shape[0], 5)
	plt.yticks(yticks_indices, labels=hz_bin_centers[yticks_indices].astype(int))
	plt.ylabel("Hz")
	plt.gca().invert_yaxis()

	plt.savefig(os.path.join(OUTPUT_DIR,plot_path))
	plt.clf()


def create_multi_signal(amplitudes, frequencies, phases, time=SINGAL_TIME, sampling_rate=8000, noise_factor=0):
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
	sample_rate, signal = scipy.io.wavfile.read(sound_file)  # File assumed to be in the same directory
	if len(signal.shape) > 1 and signal.shape[1] > 1:
		print("Using only first signal out of more")
		signal = signal[:, 0] # use only one signal if there are more than one
	print("Sginal shape: ",signal.shape)
	signal_duration = SINGAL_TIME if SINGAL_TIME is not None else len(signal) / sample_rate
	signal = signal[0:int(signal_duration * sample_rate)]  # Keep the first SINGAL_TIME seconds
	xs = np.arange(0, signal_duration, 1 / sample_rate)

	return xs, signal, sample_rate


def analyze_synthetic_signals():
	"""
	Adds various harmonic waves into a fabricate a wave form and shows the spectogram that indicates the componnens
	"""
	signal_xs, signal_ys, sample_rate, signal_name = create_multi_signal([1, 1, 1], [10, 20, 40], [0, 0, 0],
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
	# analyze_synthetic_signals()
	# plot_signal_spectogram('OSR_us_000_0010_8k.wav')
	# plot_signal_spectogram('sound_snippets/1A-T001_clap_cropped.WAV')
	plot_signal_spectogram('sound_snippets/2B-T003_doorslam.WAV')