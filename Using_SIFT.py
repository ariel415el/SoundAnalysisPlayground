import librosa
import soundfile
import matplotlib.pyplot as plt
import numpy as np

from sift_utils import pre_emphasize, split_to_frames, get_mel_filter_bank, hz2mel

NFFT = 2**12
window_size = 2**8
hop_size = 125
mel_bins = 64
mel_min_freq = 50       # Hz
mel_max_freq = 14000    # Hz


def plot_waveforms(samples_clap, samples_slam, sample_rate):
    n = samples_clap.shape[0]
    plt.plot(np.arange(n), samples_slam, color='r', label='samples_slam', alpha=0.5)
    plt.plot(np.arange(n), samples_clap, color='b', label='samples_clap', alpha=0.5)
    plt.legend()
    xticks = np.arange(0, n + n//10, n//10)
    ts = xticks / sample_rate
    plt.xticks(xticks, ts)
    plt.xlabel("Time in seconds")
    plt.ylabel("Magnitude")
    plt.savefig("Waveforms.png")


def plot_spectograms(clap_spectogram, slam_spectogram, sample_rate, type):
    freq_bins, num_frames = clap_spectogram.shape
    xticks, xlabels = get_histogram_xticks(sample_rate, num_frames, 8)
    yticks, ylabels = get_histogram_yticks(sample_rate, freq_bins, 5)
    if type=='Power':
        ylabels = np.round(ylabels / 1000, 1)
        y_title = "KHz"
    elif type=='Mel':
        ylabels = np.round(hz2mel(ylabels),1)
        y_title = "Mel"
    elif type=='Bel':
        yticks = yticks[1:]
        ylabels = ylabels[1:]
        ylabels = np.round(20 * np.log10(hz2mel(ylabels)),1)
        y_title = "Db"
    else:
        raise ValueError("No such specogram supported")
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    for i, (name, spectogram) in enumerate([("Clap", clap_spectogram), ("Slam", slam_spectogram)]):
        im = axs[i].matshow(spectogram, origin='lower', aspect='auto', cmap='jet')
        fig.colorbar(im, ax=axs[i])
        axs[i].set_title(f"{name} histogram")

        axs[i].set_xticks(xticks)
        axs[i].set_xticklabels(xlabels)
        axs[i].set_xlabel('Frame/time')
        axs[i].xaxis.set_ticks_position('bottom')

        axs[i].set_yticks(yticks)
        axs[i].set_yticklabels(ylabels)
        axs[i].set_ylabel(y_title)

    plt.tight_layout()
    plt.savefig(f"{type}-Spectogram.png")


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


def sift(samples, use_librosa=True):
    if use_librosa:
        complex_spectogran =  librosa.core.stft(
            y=samples,
            n_fft=NFFT,
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

        complex_spectogran = np.fft.rfft(frames, NFFT).T

    return complex_spectogran  # shape=(NFFT//2+1, num_frames)


def mel_filter(power_spectogram, sample_rate, use_librosa=True):
    if use_librosa:
        mel_filter_bank = librosa.filters.mel(
            sr=sample_rate,
            n_fft=NFFT,
            n_mels=mel_bins,
            fmin=mel_min_freq,
            fmax=mel_max_freq)

    else:
        mel_filter_bank, _ = get_mel_filter_bank(sample_rate, num_filters=mel_bins, nfft=NFFT,
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


def complex_to_power_spectogram(complex_spectogram):
    magnitude_spectogram = np.absolute(complex_spectogram)
    power_spectrum = (1.0 / NFFT) * (magnitude_spectogram ** 2)  # Power Spectrum: Normalizes the magnitudes by the frequency resolution
    return power_spectrum


def get_histogram_xticks(sample_rate, num_frames, num_ticks):
    FPS = sample_rate // hop_size
    # Time xticks
    x_tick_hop = num_frames // num_ticks
    xticks = np.arange(0, num_frames, x_tick_hop)
    xlabels = [f"frame {x}\n{x // FPS:.3f}s" for x in xticks]

    return xticks, xlabels


def get_histogram_yticks(sample_rate, num_freq_bins, num_ticks):
    y_tick_hop = num_freq_bins // num_ticks
    yticks = np.arange(0, num_freq_bins, y_tick_hop)
    ylabels = yticks * sample_rate/NFFT

    return yticks, ylabels


def main():
    use_librosa=True
    pre_emphasize_signal=False
    in_decibles=True
    samples_clap, samples_slam, sample_rate = load_waveforms(pre_emphasize_signal=pre_emphasize_signal)
    plot_waveforms(samples_clap, samples_slam, sample_rate)

    clap_power_spectogram = complex_to_power_spectogram(sift(samples_clap, use_librosa=use_librosa))
    slam_power_spectogram = complex_to_power_spectogram(sift(samples_slam, use_librosa=use_librosa))
    plot_spectograms(clap_power_spectogram, slam_power_spectogram, sample_rate, type='Power')

    clap_mel_spectogram = mel_filter(clap_power_spectogram, sample_rate, use_librosa=use_librosa)
    slam_mel_spectogram = mel_filter(slam_power_spectogram, sample_rate, use_librosa=use_librosa)
    plot_spectograms(clap_mel_spectogram, slam_mel_spectogram, sample_rate, type='Mel')

    clap_bel_spectogram = to_decibles(clap_mel_spectogram, use_librosa=use_librosa)
    slam_bel_spectogram = to_decibles(slam_mel_spectogram, use_librosa=use_librosa)
    plot_spectograms(clap_bel_spectogram, slam_bel_spectogram, sample_rate, type='Bel')



if __name__ == '__main__':
    main()