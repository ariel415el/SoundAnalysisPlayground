###################################################################################################
# Code inspired by https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
###################################################################################################
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
from math import ceil
COLORS=plt.get_cmap("jet")

FRAME_SIZE = 0.03
FRAME_STRIDE = 0.015
PRE_EMPHASIS = 0.97
NFFT = 512
NUM_MEL_FILTES=40
SINGAL_TIME = 10


def hz2mel(freq):
	return (2595 * np.log10(1 + freq / 700))


def mel2hz(mel):
	return (700 * (10 ** (mel / 2595) - 1))


def plot_filters(filters):
	fig = plt.figure(figsize=(20,2))
	for i,filter in enumerate(filters):
		actual_indices = np.where(filter != 0)[0]
		actual_indices = np.concatenate(([actual_indices.min() - 1], actual_indices, [actual_indices.max() + 1]))
		plt.plot(actual_indices, filter[actual_indices], color=COLORS(i /len(filters)))
	plt.legend(ncol=int(len(filters)/4), loc='lower center', bbox_to_anchor=(1.2,0.5))
	plt.show()


def plot_signals(xs, signals, names):
	fig = plt.figure(figsize=(20,2))
	for i, (x, signal,name) in enumerate(zip(xs, signals, names)):
		ax = plt.subplot(len(signals), 1, i+1)
		ax.plot(x, signal, label=name, color=COLORS(i /len(signals)))
		ax.legend(ncol=int(len(signals)/4))
	plt.show()


def plot_filters_reaction(filters_output, hz_bin_centers):
	"""

	:param filters_output: num_frames x num_filters nd_matrix
	"""
	filters_output = filters_output.T
	plt.imshow(filters_output, cmap=plt.cm.jet)

	plt.xlabel("frame number")
	yticks_indices = np.arange(0, filters_output.shape[0], 5)
	plt.yticks(yticks_indices, labels=hz_bin_centers[yticks_indices].astype(int))
	plt.ylabel("Hz")
	plt.gca().invert_yaxis()

	plt.show()


def split_to_frames(signal, frame_length, overlap_length):
	"""
	Split the signal into overlapping frames. pads the signal if necessary
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


def get_mel_filter_bank(sample_rate, num_filters=40, nfft=512):
	"""
	Creates a set of frequency filters in the mel scale
	:param sample_rate: the sample rate of the data
	:param nfft:
	:return: numpy array of shape num_filters x nfft / 22
	"""
	low_freq_mel = 0
	high_freq_mel = hz2mel(sample_rate / 2)  # Convert Hz to Mel
	mel_bins = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)  # Equally spaced in Mel scale
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
	signal = signal[0:int(SINGAL_TIME * sample_rate)]  # Keep the first 3.5 seconds
	xs = np.arange(0, SINGAL_TIME, 1 / sample_rate)

	return xs, signal, sample_rate


def pre_emphasize(signal):
	"""
	Pre ephasize increases the amplitude high frequencies and and decreases it for low frequencies.
	This adds to the balance of the signal as low frequncies usually have smaller magnitudes.
	"""
	return np.append(signal[0], signal[1:] - PRE_EMPHASIS * signal[:-1])


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
	plot_signals([signal_xs, frequencies], [signal_ys, ft], [signal_name, "ft"])


def plot_signal_spectogram(sound_file):
	"""
	Plos the spectorgram of a given sound file
	"""
	signal_xs, signal_ys, sample_rate = load_signal_from_file(sound_file)


	signal_ys_emph = pre_emphasize(signal_ys)

	# Plot signal
	plot_signals([signal_xs, signal_xs], [signal_ys, signal_ys_emph], ["origninal signal", "emphasised signal"])

	frames = split_to_frames(signal_ys_emph, int(FRAME_SIZE * sample_rate), int(FRAME_STRIDE * sample_rate))
	frames *= np.hamming(int(FRAME_SIZE * sample_rate))
	mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
	pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum: Normalizes the magnitudes by the frequency resolution

	# plot concatenated frames (with repetition)
	plot_signals([range(len(frames.reshape(-1))), range(len(mag_frames.reshape(-1))), range(len(mag_frames.reshape(-1)))],
					 [frames.reshape(-1), mag_frames.reshape(-1), pow_frames.reshape(-1)],
					 ["concatenated frames)", "concatenated mag_frames", "concatenated power_frames"])


	mel_filter_bank, hz_bin_centers = get_mel_filter_bank(sample_rate, num_filters=NUM_MEL_FILTES, nfft=NFFT)

	# Plot Mel filters
	plot_filters(mel_filter_bank)

	filter_outputs = np.dot(pow_frames, mel_filter_bank.T)
	filter_outputs = np.where(filter_outputs == 0, np.finfo(float).eps, filter_outputs)  # Numerical Stability
	filter_outputs = 20 * np.log10(filter_outputs)  # dB

	plot_filters_reaction(filter_outputs, hz_bin_centers)

if __name__ == '__main__':
	analyze_synthetic_signals()
	plot_signal_spectogram('OSR_us_000_0010_8k.wav')