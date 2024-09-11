import numpy as np
from scipy.signal import butter, lfilter, cheby1
import torch
import librosa
from utils.util import * 
# from utils.audio_filter import create_filter_block

import pyfar as pf
import pandas as pd

def renormalize(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]

#https://math.stackexchange.com/questions/970094/convert-a-linear-scale-to-a-logarithmic-scale
def convert_to_log(n, min, max, b=1):
    k = b/(np.log10(max+b) - np.log10(min+b))
    c = -1 * np.log10(min+b) * k

    return k*np.log10(b+n)+c

def renormalize_to_log(n, range1, range2, b=10):
    min1, max1 = range1
    min2, max2 = range2

    k = (max2-min2)/(np.log10(b+max1)-np.log10(b+min1))
    c = min2 - k*np.log10(b+min1)

    return k*np.log10(b+n) + c


def change_loudness(wav, loudness, loudness_meter):
    wav = pyln.normalize.peak(wav, -1.0)
    # print('Loudness before = ',loudness_meter.integrated_loudness(wav))
    wav = pyln.normalize.loudness(wav, loudness_meter.integrated_loudness(wav), loudness)

    # Hack for gain increase. Only needed with IPython.Audio. In any case PyloudNorm clips samples on gain increase.
    if is_interactive(): #see util.py
        if np.max(np.abs(wav))>1.0:
            wav = pyln.normalize.peak(wav, -1.0)
            # print('--')
            # print('Hack for gain increase for IPython. Peak norm if peaks>1.0 implemented. Loudness is = ',loudness_meter.integrated_loudness(wav))
    # print('Loudness after = ',loudness_meter.integrated_loudness(wav))
    return wav


def compress_spectrogram_simple(wav, limit=1000, hop_length=128, stft_channels=512, sample_rate=16000, loudness_meter=None, loudness=-14.0):
    source_wav_pghi = zeropad(wav, int(np.floor(len(wav)/hop_length)) * hop_length )
    source_wav_pghi_ = pghi_stft(source_wav_pghi, stft_channels=stft_channels, hop_size=hop_length)[0]
    source_wav_pghi_ = torch.from_numpy(source_wav_pghi_)

    
    fft_bins = librosa.fft_frequencies(sr=16000, n_fft=stft_channels)
    fft_bins_len = len(fft_bins)

    kernel_size = int(np.floor(fft_bins_len/np.count_nonzero(fft_bins<limit)))+1
    
    maxpool_fn = torch.nn.AvgPool2d(kernel_size=(kernel_size,1),stride=(kernel_size,1))
    
    spec_compressed = torch.zeros_like(source_wav_pghi_).fill_(torch.min(source_wav_pghi_)).unsqueeze(dim=0)
    
    spec_pooled = maxpool_fn(source_wav_pghi_.unsqueeze(0))
    spec_compressed[:,:spec_pooled.shape[1],:] = spec_pooled
    spec_compressed = spec_compressed.numpy()

    wav_compressed = pghi_istft(spec_compressed, stft_channels=stft_channels, hop_size=hop_length)
    spec_compressed = spec_compressed[0] 

    wav_compressed = change_loudness(wav_compressed, loudness, loudness_meter)
        
    return wav_compressed

def compress_spectrogram_with_centroid(wav, limit=1000, hop_length=128, stft_channels=512, sample_rate=16000, loudness_meter=None, loudness=-14.0):
    source_wav_pghi = preprocess_signal(wav)
    source_wav_pghi = zeropad(source_wav_pghi, int(np.floor(len(source_wav_pghi)/hop_length)) * hop_length )
    source_wav_pghi_ = pghi_stft(source_wav_pghi, stft_channels=stft_channels, hop_size=hop_length)[0]
    source_wav_pghi_ = torch.from_numpy(source_wav_pghi_)

    spec_centroids = librosa.feature.spectral_centroid(wav, sr=sample_rate)
    mean_spec_centroid = np.mean(spec_centroids)
    std_spec_centroid = np.std(spec_centroids)
    print("Mean Spectral Centroid = ", mean_spec_centroid)

    fft_bins = librosa.fft_frequencies(sr=16000, n_fft=stft_channels)
    fft_bins_len = len(fft_bins)

    num_bin_til_centroid = int(np.floor(np.count_nonzero(fft_bins<mean_spec_centroid)))
    num_bin_til_limit = int(np.floor(np.count_nonzero(fft_bins<limit)))
    
    kernel_size = int(np.floor(num_bin_til_centroid/(num_bin_til_limit)))+1
    print(fft_bins_len, num_bin_til_centroid, num_bin_til_limit, kernel_size) 

    maxpool_fn = torch.nn.AvgPool2d(kernel_size=(kernel_size,1),stride=(kernel_size,1))
    
    spec_compressed = torch.zeros_like(source_wav_pghi_).fill_(torch.min(source_wav_pghi_)).unsqueeze(dim=0)

    source_wav_pghi_ = source_wav_pghi_[:num_bin_til_centroid,:]
    spec_pooled = maxpool_fn(source_wav_pghi_.unsqueeze(0))
    spec_compressed[:,:spec_pooled.shape[1],:] = spec_pooled
    spec_compressed = spec_compressed.numpy()

    wav_compressed = pghi_istft(spec_compressed, stft_channels=stft_channels, hop_size=hop_length)
    spec_compressed = spec_compressed[0]

    wav_compressed = change_loudness(wav_compressed, loudness, loudness_meter)
        
    return wav_compressed


def pitch_shift_centroid(wav, limit=1000, sample_rate=16000, loudness_meter=None, loudness=-14.0):
    spec_centroids = librosa.feature.spectral_centroid(wav, sr=sample_rate)
    mean_spec_centroid = np.mean(spec_centroids)
    

    num_steps = 12 * np.log2(mean_spec_centroid/limit)
    pitch_shifted_wav = librosa.effects.pitch_shift(wav, sr=sample_rate, n_steps=-1*num_steps)
    print("Mean Spectral Centroid = ", mean_spec_centroid, " Pitch Shift Steps = ", num_steps)

    wav_compressed = change_loudness(pitch_shifted_wav, loudness, loudness_meter)
    print("Mean Spectral Centroid = ", mean_spec_centroid, " Pitch Shift Steps = ", num_steps, " Final Shifter Centroid = ", np.mean(librosa.feature.spectral_centroid(wav_compressed, sr=sample_rate)))
        
    return wav_compressed


##
## Old Method
##
# def equalize_audio(wav, sample_rate=16000, loudness_meter=None, loudness=-14.0):
#     eq = create_filter_block(sample_rate)
#     eq.reset()
#     audio = eq.process_block(wav)
#     audio /= np.max(np.abs(audio))
#     if loudness_meter is not None:
#         audio = change_loudness(audio, loudness, loudness_meter)
#     return audio


def equalize_audio(wav, type="BLACK", sample_rate=16000, minval=None, maxval=None, freq_response_dir=''):#, loudness_meter=None, loudness=-14.0):

    freq_response_file = ''
    if type == "BLACK":
        freq_response_file = freq_response_dir+'smooth_rollAvg6_freq-response-DRAKE-BLACK-RAW-0.75-1722943603.275152.csv'
    elif type == "YELLOW":
        freq_response_file = freq_response_dir+'smooth_rollAvg6_freq-response-DRAKE-YELLOW-RAW-0.75-1722942712.900473.csv'
    elif type == "RED":
        freq_response_file = freq_response_dir+'smooth_rollAvg6_freq-response-DRAKE-RED-RAW-0.75-1722937125.680299.csv'
    elif type == "WHITE":
        freq_response_file = freq_response_dir+'smooth_rollAvg6_freq-response-DRAKE-WHITE-RAW-0.75-1722941149.946043.csv'

    print(freq_response_file)
        
    df = pd.read_csv(freq_response_file)
    new_vals = []

    if minval is None:
        minval = np.min(df['smoothed_accVals'])
    if maxval is None:
        maxval = np.max(df['smoothed_accVals'])
    for i in df['smoothed_accVals']:
        # print(i)
        # new_vals.append(-1*renormalize_to_log(i, (np.min(df['smoothed_accVals']), np.max(df['smoothed_accVals'])),(0,1)))
        new_vals.append(-1*renormalize_to_log(i, (minval, maxval),(0,1)))

    
    y = pf.classes.audio.Signal(wav, sampling_rate=sample_rate)
    for ind, freq_c in enumerate(df['freqVals']):
        y = pf.dsp.filter.bell(signal=y, center_frequency=freq_c, gain=new_vals[ind], quality=4)
        
    audio = y._data[0]/np.max(np.abs(y._data[0]))
    # if loudness_meter is not None:
    #     audio = change_loudness(audio, loudness, loudness_meter)
    return audio


def cheby_lowpass(data, lowcut, fs, order=5, btype='lowpass'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    rp = 0.0001
    b, a = cheby1(order, rp, low)

    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5,btype='bandpass'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype=btype)
    return b, a

def butter_lowhighpass(cut, fs, order=5, btype='lowpass'):
    nyq = 0.5 * fs
    cut = cut / nyq
    b, a = butter(order, cut, btype=btype)
    return b, a

def butter_bandpass_filter(data, highcut, fs,lowcut=None,  order=5, btype='bandpass'):
    if btype=='bandpass' or btype=='bandstop':
        b, a = butter_bandpass(lowcut, highcut, fs, order=order, btype=btype)
    else:
        b, a = butter_lowhighpass(highcut, fs, order=order, btype=btype)
    y = lfilter(b, a, data)
    return y

# I eventually did not use this one. 
#https://github.com/mayank12gt/Audio-Equalizer/blob/master/equalizer.py
def butter_bandpass_filter_withgain(audio, lowcut, highcut, fs, order=5, btype='bandpass', gain=1): 
    b, a = butter_bandpass(lowcut, highcut, fs, order=order, btype=btype)
    b = b * gain
    y = lfilter(b, a, audio)
    return y


def applyFBFadeFilter(forward_fadetime,backward_fadetime,signal,fs,expo=1):
    forward_num_fad_samp = int(forward_fadetime*fs) 
    backward_num_fad_samp = int(backward_fadetime*fs) 
    signal_length = len(signal) 
    fadefilter = np.ones(signal_length)
    if forward_num_fad_samp>0:
        fadefilter[0:forward_num_fad_samp]=np.linspace(0,1,forward_num_fad_samp)**expo
    if backward_num_fad_samp>0:
        fadefilter[signal_length-backward_num_fad_samp:signal_length]=np.linspace(1,0,backward_num_fad_samp)**expo
    return fadefilter*signal
