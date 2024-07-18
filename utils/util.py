import numpy as np
import IPython
from IPython.display import Audio, display
import librosa, librosa.display
import soundfile as SF
import noisereduce as nr
from scipy.signal import butter, lfilter

from torch_pitch_shift import pitch_shift


import sys, random  
sys.path.insert(0, '../')
from audioldm2 import text_to_audio, build_model, seed_everything

import pyloudnorm as pyln


from tifresi.utils import load_signal
from tifresi.utils import preprocess_signal
from tifresi.transforms import log_spectrogram
from tifresi.transforms import inv_log_spectrogram
from tifresi.stft import GaussTF, GaussTruncTF

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#--------------------------------------------------------#
#-----------------STFT related---------------------------#
#--------------------------------------------------------#
def zeropad(signal, audio_length):
    if len(signal) < audio_length:
        return np.append(
            signal, 
            np.zeros(audio_length - len(signal))
        )
    else:
        signal = signal[0:audio_length]
        return signal
    
def pghi_stft(x, hop_size=128, stft_channels=512, use_truncated_window=True):
    if use_truncated_window:
        stft_system = GaussTruncTF(hop_size=hop_size, stft_channels=stft_channels)
    else:
        stft_system = GaussTF(hop_size=hop_size, stft_channels=stft_channels)
    Y = stft_system.spectrogram(x)
    log_Y= log_spectrogram(Y)
    return np.expand_dims(log_Y,axis=0)

def pghi_istft(x, hop_size=128, stft_channels=512, use_truncated_window=True):
    if use_truncated_window:
        stft_system = GaussTruncTF(hop_size=hop_size, stft_channels=stft_channels)
    else:
        stft_system = GaussTF(hop_size=hop_size, stft_channels=stft_channels)

    x = np.squeeze(x,axis=0)
    new_Y = inv_log_spectrogram(x)
    new_y = stft_system.invert_spectrogram(new_Y)
    return new_y

#--------------------------------------------------------#
#-----------------Plots related--------------------------#
#--------------------------------------------------------#

def plot_all(wavs, titles):
    num_cols = 4
    
    nrows = len(wavs)//num_cols 
    if len(wavs)%num_cols > 0:
        nrows += 1
    print('num rows=', nrows)
    fig, axs = plt.subplots(nrows=nrows, ncols=num_cols, figsize=(25,3))
    for i, wav in enumerate(wavs):
        nrow = i//num_cols
        ncol = i%num_cols
        IPython.display.display(IPython.display.Audio(wav, rate=16000, normalize=False))

        D = librosa.amplitude_to_db(np.abs(librosa.stft(wav, hop_length=512)),ref=np.max)
        if nrows>1:
            librosa.display.specshow(D, y_axis='linear', sr=16000, hop_length=512, x_axis='time', ax=axs[nrow][ncol])
            axs[nrow][ncol].set_title(titles[i])
            divider = make_axes_locatable(axs[ncol])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            
        else:
            im = librosa.display.specshow(D, y_axis='linear', sr=16000, hop_length=512, x_axis='time', ax=axs[ncol])
            axs[ncol].set_title(titles[i])
            divider = make_axes_locatable(axs[ncol])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

def plot_single(wav, title):
    IPython.display.display(IPython.display.Audio(wav, rate=16000, normalize=False))

    D = librosa.amplitude_to_db(np.abs(librosa.stft(wav, hop_length=512)),ref=np.max)
    librosa.display.specshow(D, y_axis='linear', sr=16000, hop_length=512, x_axis='time')
    plt.title(title)


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')