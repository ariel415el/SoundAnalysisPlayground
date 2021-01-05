import numpy as np
from matplotlib import pyplot as plt

from utils.sound_utils import hz2mel, mel2hz

COLORS=plt.get_cmap("jet")


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


def plot_spectograms(clap_spectogram, slam_spectogram, sample_rate, hop_size, nfft=None, mel_min_freq=None, mel_max_freq=None, type="Power"):
    freq_bins, num_frames = clap_spectogram.shape
    xticks, xlabels = get_histogram_xticks(num_frames, sample_rate / hop_size, num_ticks=8)
    if type=='Power':
        yticks, ylabels = get_histogram_yticks(sample_rate, freq_bins, nfft, mel_min_freq, mel_max_freq, num_ticks=5)
        ylabels /= 1000
        y_title = "KHz"
    elif type=='Mel':
        yticks, ylabels = get_histogram_yticks(sample_rate, freq_bins, nfft, mel_min_freq, mel_max_freq, num_ticks=5, type='mel')
        y_title = "Mel"
    elif type=='Bel':
        yticks, ylabels = get_histogram_yticks(sample_rate, freq_bins, nfft, mel_min_freq, mel_max_freq, num_ticks=5, type='mel')
        yticks = yticks[1:]
        ylabels = ylabels[1:]
        ylabels = 20 * np.log10(hz2mel(ylabels))
        y_title = "Db"
    else:
        raise ValueError("No such specogram supported")

    ylabels = np.round(ylabels, 1)
    ylabels = [f"{l} {y_title}\nbin {i}" for i, l in zip(yticks, ylabels)]
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

    plt.title(f"{type}-Spectogram.png")
    plt.tight_layout()
    plt.savefig(f"{type}-Spectogram.png")


def get_histogram_xticks(num_frames, fps, num_ticks):
    # Time xticks
    x_tick_hop = num_frames // num_ticks
    xticks = np.arange(0, num_frames, x_tick_hop)
    xlabels = [f"frame {x}\n{x / fps:.2f}s" for x in xticks]

    return xticks, xlabels


def get_histogram_yticks(sample_rate, num_freq_bins, nfft, mel_min_freq, mel_max_freq, num_ticks, type='hz'):
    y_tick_hop = num_freq_bins // num_ticks
    yticks = np.arange(0, num_freq_bins, y_tick_hop)
    if type == 'hz':
        ylabels = yticks * sample_rate/nfft
    else:
        mel_bins = mel2hz(np.linspace(hz2mel(mel_min_freq), hz2mel(mel_max_freq), num_freq_bins + 2)[1:-1])  # Equally spaced in Mel scale
        ylabels = mel_bins[::y_tick_hop]

    return yticks, ylabels


def plot_filters(filters, plot_path):
    plt.figure(figsize=(20,2))
    for i,filter in enumerate(filters):
        actual_indices = np.where(filter != 0)[0]
        actual_indices = np.concatenate(([actual_indices.min() - 1], actual_indices, [actual_indices.max() + 1]))
        plt.plot(actual_indices, filter[actual_indices], color=COLORS(i /len(filters)))
    plt.legend(ncol=int(len(filters)/4), loc='lower center', bbox_to_anchor=(1.2,0.5))
    plt.savefig(plot_path)
    plt.clf()


def plot_signals(xs, signals, names, plot_path):
    plt.figure(figsize=(20,2))
    for i, (x, signal,name) in enumerate(zip(xs, signals, names)):
        ax = plt.subplot(len(signals), 1, i+1)
        ax.plot(x, signal, label=name, color=COLORS(i /len(signals)))
        ax.legend(ncol=int(len(signals)/4))
    plt.savefig(plot_path)
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

    plt.savefig(plot_path)
    plt.clf()