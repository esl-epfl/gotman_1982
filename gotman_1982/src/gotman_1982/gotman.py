import numpy as np
from scipy import signal
from scipy.interpolate import interp1d


def removal_60hz(X_data, fs):
    """
    Remove 60Hz noise from EEG signal
    Args:
    X_data: EEG data, shape (n_samples, n_channels)
    fs: sampling frequency
    Returns:
    X_filtered: filtered data, shape (n_samples, n_channels)
    """
    f0 = 60
    Q = 30
    w0 = f0 / (fs / 2)
    w0 = f0
    b, a = signal.iirnotch(w0, Q, fs=fs)
    X_filtered = signal.filtfilt(b, a, X_data)
    return X_filtered


def lowpass_filter(X_data, fs, cutoff=60):
    """
    Lowpass filter for EEG signal
    Args:
    X_data: EEG data, shape (n_samples, n_channels)
    fs: sampling frequency
    cutoff: cutoff frequency
    Returns:
    X_filtered: filtered data, shape (n_samples, n_channels)
    """
    b, a = signal.butter(4, cutoff, fs=fs, btype="low", analog=False)
    X_filtered = signal.filtfilt(b, a, X_data)
    return X_filtered


def half_wave(t, X_filtered, prominence=10):
    """
    Convert signal to half waves
    Args:
    t: array of time points
    X_filtered: filtered EEG data, shape (n_samples, n_channels)
    prominence: prominence for peak detection
    Returns:
    X_halfwaves: half wave transformation of data, shape (n_samples, n_channels)
    """
    maxs, _ = signal.find_peaks(X_filtered, prominence=prominence)
    # same for minimums
    mins, _ = signal.find_peaks(-X_filtered, prominence=prominence)
    peaks = np.concatenate((maxs, mins))
    if len(peaks) == 0:
        return X_filtered
    # 1d interpolation of peaks
    f_halfwaves = interp1d(t[peaks], X_filtered[peaks], fill_value="extrapolate")
    X_halfwaves = f_halfwaves(t)
    return X_halfwaves


def calculate_period(t, X_halfwaves):
    """
    Calculate period of half waves
    Args:
    t: array of time points
    X_halfwaves: half wave transformation of data, shape (n_samples, n_channels)
    Returns:
    period: array of periods of half waves
    """
    peaks, _ = signal.find_peaks(X_halfwaves)
    if len(peaks) <= 1:
        return np.array([t[-1] - t[0]])
    period = np.zeros(len(peaks) - 1)
    for i in range(1, len(peaks)):
        period[i - 1] = t[peaks[i]] - t[peaks[i - 1]]
    return period


def gotman_algorithm(
    t, eeg_data, fs, window_size=2, overlap=0.5, background_size=16, transition_size=12
):
    """
    Gotman algorithm for seizure detection
    Args:
    t: array of time points
    eeg_data: EEG data, shape (n_samples, n_channels)
    fs: sampling frequency
    window_size: window size for detection in seconds
    overlap: overlap between windows [%]
    background_size: size of background window in seconds
    transition_size: size of transition window in seconds
    Returns:
    seizure_detections: boolean array of seizure detections, shape (n_samples,)
    """
    window_size = int(window_size * fs)
    step = int(window_size * (1 - overlap))
    background_size = int(background_size * fs)
    transition_size = int(transition_size * fs)
    channel_detection = np.zeros(eeg_data.shape)
    idx_detection = [[] for _ in range(eeg_data.shape[0])]

    for k, eeg_channel in enumerate(eeg_data):
        X_data = eeg_channel
        # preprocessing
        X_filtered = lowpass_filter(X_data, fs)

        # conversion to half waves according to Gotman algorithm
        X_halfwave = half_wave(t, X_filtered)

        # channel detection
        for i in range(background_size + transition_size, X_halfwave.shape[0], step):
            background_amp = np.mean(
                np.abs(
                    X_halfwave[
                        i - (background_size + transition_size) : i - transition_size
                    ]
                )
            )
            window_amp = np.mean(np.abs(X_halfwave[i : i + window_size]))
            period_halfwaves = calculate_period(
                t[i : i + window_size], X_halfwave[i : i + window_size]
            )
            mean_period = np.mean(period_halfwaves)

            mean_period_bool = (mean_period > 0.025) & (
                mean_period < 0.150
            )  # between 3 and 20 Hz
            variation_coeff = np.std(period_halfwaves) / mean_period
            variation_coeff_bool = variation_coeff < 0.6
            if (
                (window_amp > 3 * background_amp)
                and mean_period_bool
                and variation_coeff_bool
            ):
                channel_detection[k, i : min(channel_detection.shape[1], i + step)] = 1
                idx_detection[k].append(i)

    # seizure detection criteria
    seizure_detections = np.zeros(eeg_data.shape[1])
    for channel in range(len(idx_detection)):
        for i in range(len(idx_detection[channel]) - 1):
            # check adjacent epochs within same channel
            if (
                np.abs(idx_detection[channel][i + 1] - idx_detection[channel][i])
                == step
            ):
                seizure_detections[
                    idx_detection[channel][i] : idx_detection[channel][i + 1]
                ] = 1
            for channel2 in range(len(idx_detection)):
                if channel == channel2:
                    continue
                for j in range(len(idx_detection[channel2])):
                    # check adjacent epochs within different channels
                    if idx_detection[channel2][j] - idx_detection[channel][i] == step:
                        seizure_detections[
                            idx_detection[channel][i] : idx_detection[channel2][j]
                        ] = 1
                    # check same epoch within different channels
                    if idx_detection[channel2][j] == idx_detection[channel][i]:
                        seizure_detections[
                            idx_detection[channel][i] : idx_detection[channel][i] + step
                        ] = 1

    return seizure_detections
