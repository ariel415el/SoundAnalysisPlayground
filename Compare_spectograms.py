import soundfile
import os
import numpy as np
from utils.fft_utils import pre_emphasize, complex_to_power_spectogram, sift
from utils.plot_utils import plot_waveforms, plot_spectograms
from utils.sound_utils import mel_filter, to_decibles

NFFT = 2**11
window_size = 1000
hop_size = 500
mel_bins = 64
mel_min_freq = 0       # Hz
mel_max_freq = 14000    # Hz


def load_waveforms(file_paths,  pre_emphasize_signal=False):
    waveform_list, sample_rate_list, name_list = [], [], []
    for file_path in file_paths:
        samples, sample_rate, = soundfile.read(file_path)
        samples = samples[:, 0]
        if pre_emphasize_signal:
            samples = pre_emphasize(samples)
        waveform_list.append(samples)
        sample_rate_list.append(sample_rate)
        name_list.append(os.path.basename(file_path))

    assert len(np.unique(len(x) for x in waveform_list)) == 1, "Make sure all audio files have the same length"
    assert len(np.unique(sample_rate_list)) == 1, "Make sure all audio files have the same sample rate"

    return waveform_list, name_list, sample_rate_list[0]


def main():
    use_librosa=True
    pre_emphasize_signal=False
    file_paths = ["/home/ariel/projects/sound/SoundEventDetection-Pytorch/samples/Meron_S017-S002T2_2_car_door_37s.WAV",
                  "/home/ariel/projects/sound/SoundEventDetection-Pytorch/samples/Meron_S017-S002T2_car_door_14s.WAV",
                  "/home/ariel/projects/sound/SoundEventDetection-Pytorch/samples/Meron_S017-S002T2_clap_9s.WAV"
                  ]

    waveform_list, names_list, sample_rate = load_waveforms(file_paths, pre_emphasize_signal=pre_emphasize_signal)
    plot_waveforms(waveform_list, names_list, sample_rate)

    complex_spectograms = [sift(waveform, window_size, hop_size, NFFT, use_librosa=use_librosa)
                           for waveform in waveform_list]

    power_spectograms = [complex_to_power_spectogram(spectogram, NFFT)
                         for spectogram in complex_spectograms]
    plot_spectograms(power_spectograms, names_list, sample_rate, hop_size,
                     nfft=NFFT, type='Power')

    mel_spectograms = [mel_filter(spectogram, NFFT, mel_bins, mel_min_freq, mel_max_freq, sample_rate, use_librosa=use_librosa)
                        for spectogram in power_spectograms]
    plot_spectograms(mel_spectograms, names_list, sample_rate, hop_size,
                     mel_min_freq=mel_min_freq, mel_max_freq=mel_max_freq, type='Mel')

    bel_spectograms = [to_decibles(spectogram, NFFT)
                       for spectogram in mel_spectograms]
    plot_spectograms(bel_spectograms, names_list, sample_rate, hop_size,
                     mel_min_freq=mel_min_freq, mel_max_freq=mel_max_freq, type='Bel')


if __name__ == '__main__':
    main()