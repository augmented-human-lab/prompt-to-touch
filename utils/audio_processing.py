import torch
import librosa
from utils.util import * 

def change_loudness(wav, loudness, loudness_meter):
    wav = pyln.normalize.peak(wav, -1.0)
    print('Loudness before = ',loudness_meter.integrated_loudness(wav))
    wav = pyln.normalize.loudness(wav, loudness_meter.integrated_loudness(wav), loudness)
    print('Loudness after = ',loudness_meter.integrated_loudness(wav))


    # Hack for gain increase. Only needed with IPython.Audio. In any case PyloudNorm clips samples on gain increase.
    if is_interactive(): #see util.py
        if np.max(np.abs(wav))>1.0:
            wav = pyln.normalize.peak(wav, -1.0)
            print('Hack for gain increase for IPython. Peak norm if peaks>1.0 implemented. Loudness is = ',loudness_meter.integrated_loudness(wav))
    
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
    source_wav_pghi = zeropad(wav, int(np.floor(len(wav)/hop_length)) * hop_length )
    source_wav_pghi_ = pghi_stft(source_wav_pghi, stft_channels=stft_channels, hop_size=hop_length)[0]
    source_wav_pghi_ = torch.from_numpy(source_wav_pghi_)

    spec_centroids = librosa.feature.spectral_centroid(wav, sr=sample_rate)
    mean_spec_centroid = np.mean(spec_centroids)
    std_spec_centroid = np.std(spec_centroids)
    print("Mean Spectral Centroid = ", mean_spec_centroid, np.std(spec_centroids))

    fft_bins = librosa.fft_frequencies(sr=16000, n_fft=stft_channels)
    fft_bins_len = len(fft_bins)

    num_bin_til_centroid = int(np.floor(np.count_nonzero(fft_bins<mean_spec_centroid)))
    num_bin_til_limit = int(np.floor(np.count_nonzero(fft_bins<limit)))
    
    kernel_size = int(np.floor(num_bin_til_centroid/(num_bin_til_limit)))+1

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