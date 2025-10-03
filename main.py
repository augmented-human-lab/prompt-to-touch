import librosa
import numpy as np
import pyloudnorm as pyln
from openai import OpenAI
import soundfile as sf
import noisereduce as nr

import sys, os  
sys.path.insert(0, '../')
import utils
from utils.audio_generation import sample, get_model, sample_multiple
from utils.audio_processing import compress_spectrogram_simple, compress_spectrogram_with_centroid, \
equalize_audio, butter_bandpass_filter, pitch_shift_centroid,change_loudness, cheby_lowpass

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Important.

model_name = 'audioldm2-full' # Larger model; More GPU memory ~[12-15] GB; 

#Audio params
loudness_dbfs = -20.0
loudness = -20.0
sample_rate = 16000
stft_channels = 1024
hop_length = 128

# Diffusion params
guidance_scale = 3
n_candidates = 1
batch_size = 1
ddim_steps = 100

loudness_meter = pyln.Meter(sample_rate)

save_folder = 'output_dir/'
os.makedirs(save_folder, exist_ok=True)

'''
Foley Interpreter
'''
def foley_interpreter(txt, client):
    content = "Your job is to come up with a description of an audio effect for the description of the haptic touch experience. That means you describe an audio effect that is likely to exist that could resemble the key characteristics of the described haptic touch experience. If the description contains non audible aspects, try imagine what audio effects may result in the haptic experience. Here is the description of the haptic touch experience: '"+txt.lower()+"'. How would you describe the translated audio effect? Think step by step. If applicable and meaningful to the sound, describe how the sound effect evolves step by step. Keep in mind that only one speaker is available to play the effect. "

    response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
          {"role": "system", "content": "You are an expert haptic feedback designer."},
          {"role": "user", "content": content}
      ]
    )

    initial_response = response.choices[0].message.content
    response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {"role": "system", "content": "You are an expert in writing audio captions for sound effects."},
        {"role": "user", "content": content},
        {"role": "assistant", "content": initial_response},
        {"role": "user", "content": "Now output the final audio effect description in the format of a short audio caption. The effect should be unique and highlight the key characteristic of the haptic description. If the sound effect evolves over time, only focus on the the most important step part that is most characteristic for the sound effect. Ensure that the caption is in double-quotes. The audio caption should be short, and descriptive, focusing on non-technical descriptions of a sound effect. Ignore background noise and avoid humming."}
      ]
    )

    foley_language_phrase = response.choices[0].message.content
    print(foley_language_phrase)

    return foley_language_phrase


'''
Audio Generator
'''
def audio_generator(foley_language_phrase, latent_diffusion, random_seed):
    audio = sample_multiple(latent_diffusion, foley_language_phrase, n_candidates=3, ddim_steps=100, guidance_scale=3.0, \
             random_seed=random_seed)

    return audio[0][0][0]


def generate_haptic_effect_samples(prompt, latent_diffusion, ai_client, sample_time_s=3, output_dir_name="tmp", equalizer_profile=None, sample_types=['foley_compression']): # sample_types=['basic_none','basic_ps','basic_compression','foley_none','foley_ps','foley_compression']
    random_num = np.random.randint(0,65000)
    output_dir_name = output_dir_name.lower().replace("/","_")
    os.makedirs(save_folder+output_dir_name, exist_ok=True)

    # 1. Get foley phrase
    foley_language_phrase = foley_interpreter(prompt, ai_client)
    
    #2. Trim to 3 seconds: Find onsets - so that we can trim to 3 secs (from 10 sec files)
    if "basic_none" in sample_types or "basic_ps" in sample_types or "basic_compression" in sample_types:
        print("Generating audio without foley...")
        audio_without_foley = audio_generator(prompt, latent_diffusion, random_num)
        audio_without_foley_nr = nr.reduce_noise(y=audio_without_foley, sr=sample_rate)
        audio_without_foley_onset_times = librosa.frames_to_time(librosa.onset.onset_detect(y=audio_without_foley_nr, sr=sample_rate))
        start2 = int(audio_without_foley_onset_times[0]*sample_rate)
        audio_without_foley_3secs = audio_without_foley[start2: start2+sample_time_s*sample_rate]

        #1 basic_none
        if "basic_none" in sample_types:
            basic_none = audio_without_foley_3secs
            basic_none = butter_bandpass_filter(basic_none, highcut=1000, fs=sample_rate, lowcut=None, order=17, btype='lowpass')
            if equalizer_profile:
                print("Equalizing with profile: ", equalizer_profile)
                basic_none = equalize_audio(basic_none, equalizer_profile=equalizer_profile, sample_rate=sample_rate, minval=None, maxval=None)
            else:
                print("No Equalization applied.")
            basic_none = change_loudness(basic_none, loudness, loudness_meter)
            print('Original Loudness basic_none= ',loudness_meter.integrated_loudness(basic_none))
            sf.write(save_folder+output_dir_name+"/"+"basic_none.wav", basic_none, samplerate=sample_rate)

        #2 basic_ps
        if "basic_ps" in sample_types:
            basic_ps = audio_without_foley_3secs
            basic_ps = pitch_shift_centroid(basic_ps, limit=250, sample_rate=sample_rate, loudness_meter=loudness_meter, loudness=loudness)
            basic_ps = butter_bandpass_filter(basic_ps, highcut=1000, fs=sample_rate, lowcut=None, order=17, btype='lowpass')
            if equalizer_profile:
                print("Equalizing with profile: ", equalizer_profile)
                basic_ps = equalize_audio(basic_ps, equalizer_profile=equalizer_profile, sample_rate=sample_rate, minval=None, maxval=None)
            else:
                print("No Equalization applied.")
            basic_ps = change_loudness(basic_ps, loudness, loudness_meter)
            sf.write(save_folder+output_dir_name+"/"+"basic_ps.wav", basic_ps, samplerate=sample_rate)
    
        #3 basic_compression
        if "basic_compression" in sample_types:
            basic_compression = audio_without_foley_3secs
            basic_compression = compress_spectrogram_with_centroid(basic_compression, limit=1000, hop_length=hop_length, stft_channels=stft_channels, \
                                                                sample_rate=sample_rate, loudness_meter=loudness_meter, loudness=loudness)
            basic_compression = butter_bandpass_filter(basic_compression, highcut=1000, fs=sample_rate, lowcut=None, order=17, btype='lowpass')
            if equalizer_profile:
                print("Equalizing with profile: ", equalizer_profile)
                basic_compression = equalize_audio(basic_compression, equalizer_profile=equalizer_profile, sample_rate=sample_rate, minval=None, maxval=None)
            else:
                print("No Equalization applied.")
            basic_compression = change_loudness(basic_compression, loudness, loudness_meter)
            sf.write(save_folder+output_dir_name+"/"+"basic_compression.wav", basic_compression, samplerate=sample_rate)
        
    
    if "foley_none" in sample_types or "foley_ps" in sample_types or "foley_compression" in sample_types:
            audio_with_foley = audio_generator(foley_language_phrase, latent_diffusion, random_num)
            audio_with_foley_nr = nr.reduce_noise(y=audio_with_foley, sr=sample_rate)
            audio_with_foley_onset_times = librosa.frames_to_time(librosa.onset.onset_detect(y=audio_with_foley_nr, sr=sample_rate))
            start1 = int(audio_with_foley_onset_times[0]*sample_rate)
            audio_with_foley_3secs = audio_with_foley[start1: start1+sample_time_s*sample_rate]
            
            #4 foley_none
            if "foley_none" in sample_types:
                foley_none = audio_with_foley_3secs
                foley_none = butter_bandpass_filter(foley_none, highcut=1000, fs=sample_rate, lowcut=None, order=17, btype='lowpass')
                if equalizer_profile:
                    print("Equalizing with profile: ", equalizer_profile)
                    foley_none = equalize_audio(foley_none, equalizer_profile=equalizer_profile, sample_rate=sample_rate, minval=None, maxval=None)
                else:
                    print("No Equalization applied.")
                foley_none = change_loudness(foley_none, loudness, loudness_meter)
                print('Original Loudness foley_none= ',loudness_meter.integrated_loudness(foley_none))
                sf.write(save_folder+output_dir_name+"/"+"foley_none.wav", foley_none, samplerate=sample_rate)
    
            
            #5 foley_ps
            if "foley_ps" in sample_types:
                foley_ps = audio_with_foley_3secs
                foley_ps = pitch_shift_centroid(foley_ps, limit=250, sample_rate=sample_rate, loudness_meter=loudness_meter, loudness=loudness)
                foley_ps = butter_bandpass_filter(foley_ps, highcut=1000, fs=sample_rate, lowcut=None, order=17, btype='lowpass')
                if equalizer_profile:
                    print("Equalizing with profile: ", equalizer_profile)
                    foley_ps = equalize_audio(foley_ps, equalizer_profile=equalizer_profile, sample_rate=sample_rate, minval=None, maxval=None)
                else:
                    print("No Equalization applied.")
                foley_ps = change_loudness(foley_ps, loudness, loudness_meter)
                sf.write(save_folder+output_dir_name+"/"+"foley_ps.wav", foley_ps, samplerate=sample_rate)
    
            #6 foley_compression
            if "foley_compression" in sample_types:
                foley_compression = audio_with_foley_3secs
                foley_compression = compress_spectrogram_with_centroid(foley_compression, limit=1000, hop_length=hop_length, stft_channels=stft_channels, \
                                                                    sample_rate=sample_rate, loudness_meter=loudness_meter, loudness=loudness)
                foley_compression = butter_bandpass_filter(foley_compression, highcut=1000, fs=sample_rate, lowcut=None, order=17, btype='lowpass')
                if equalizer_profile:
                    print("Equalizing with profile: ", equalizer_profile)
                    foley_compression = equalize_audio(foley_compression, equalizer_profile=equalizer_profile, sample_rate=sample_rate, minval=None, maxval=None)
                else:
                    print("No Equalization applied.")
                foley_compression = change_loudness(foley_compression, loudness, loudness_meter)
                sf.write(save_folder+output_dir_name+"/"+"foley_compression.wav", foley_compression, samplerate=sample_rate)

def main(prompt):
    # Load environment variables from .env file if it exists
    env_file_path = 'config/.env'
    if os.path.exists(env_file_path):
        with open(env_file_path, 'r') as fh:
            vars_dict = {}
            for line in fh.readlines():
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    vars_dict[key.strip()] = value.strip()
            
            os.environ.update(vars_dict)

    latent_diffusion = get_model(model_name)
    ai_client = OpenAI()

    generate_haptic_effect_samples(prompt, latent_diffusion, ai_client)

if __name__ == '__main__':
    prompt=sys.argv[1] 
    main(prompt)    