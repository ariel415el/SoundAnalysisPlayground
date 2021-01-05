import soundfile

from utils.fft_utils import pre_emphasize, complex_to_power_spectogram, sift
from utils.plot_utils import plot_waveforms, plot_spectograms
from utils.sound_utils import mel_filter, to_decibles

NFFT = 2**10
window_size = 2**9
hop_size = 2**8
mel_bins = 64
mel_min_freq = 0       # Hz
mel_max_freq = 14000    # Hz


def load_waveforms(pre_emphasize_signal=False):
    samples_clap, sample_rate_clap = soundfile.read("sound_snippets/1A-T001_clap_cropped.WAV")
    samples_slam, sample_rate_slam = soundfile.read("sound_snippets/2B-T003_doorslam.WAV")
    samples_clap = samples_clap[:, 0]
    samples_slam = samples_slam[:, 0]
    assert(sample_rate_slam == sample_rate_clap)
    assert(samples_clap.shape == samples_slam.shape)

    if pre_emphasize_signal:
        samples_clap = pre_emphasize(samples_clap)
        samples_slam = pre_emphasize(samples_slam)

    return samples_clap, samples_slam, sample_rate_clap


def main():
    use_librosa=True
    pre_emphasize_signal=True

    samples_clap, samples_slam, sample_rate = load_waveforms(pre_emphasize_signal=pre_emphasize_signal)
    plot_waveforms(samples_clap, samples_slam, sample_rate)

    clap_complex_histogram = sift(samples_clap, window_size, hop_size, NFFT, use_librosa=use_librosa)
    slam_complex_histogram = sift(samples_slam, window_size, hop_size, NFFT, use_librosa=use_librosa)

    clap_power_spectogram = complex_to_power_spectogram(clap_complex_histogram, NFFT)
    slam_power_spectogram = complex_to_power_spectogram(slam_complex_histogram, NFFT)
    plot_spectograms(clap_power_spectogram, slam_power_spectogram, sample_rate, hop_size, nfft=NFFT, type='Power')

    clap_mel_spectogram = mel_filter(clap_power_spectogram, NFFT, mel_bins, mel_min_freq, mel_max_freq, sample_rate, use_librosa=use_librosa)
    slam_mel_spectogram = mel_filter(slam_power_spectogram, NFFT, mel_bins, mel_min_freq, mel_max_freq, sample_rate, use_librosa=use_librosa)
    plot_spectograms(clap_mel_spectogram, slam_mel_spectogram, sample_rate, hop_size, mel_min_freq=mel_min_freq, mel_max_freq=mel_max_freq, type='Mel')

    clap_bel_spectogram = to_decibles(clap_mel_spectogram, use_librosa=use_librosa)
    slam_bel_spectogram = to_decibles(slam_mel_spectogram, use_librosa=use_librosa)
    plot_spectograms(clap_bel_spectogram, slam_bel_spectogram, sample_rate, hop_size, mel_min_freq=mel_min_freq, mel_max_freq=mel_max_freq, type='Bel')


if __name__ == '__main__':
    main()