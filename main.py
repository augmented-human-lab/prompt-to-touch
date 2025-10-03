import numpy as np
import pyloudnorm as pyln
from openai import OpenAI
import soundfile as sf

import sys, os  
sys.path.insert(0, '../')
import utils
from utils.audio_generation import sample, get_model
from utils.audio_processing import compress_spectrogram_simple, compress_spectrogram_with_centroid

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Important.

model_name = 'audioldm_16k_crossattn_t5' # Smaller model; Less GPU memory ~[6-9]GB; 
# model_name = 'audioldm2-full' # Larger model; More GPU memory ~[12-15] GB; 

#Audio params
loudness_dblufs = -10.0
sample_rate = 16000
stft_channels = 1024
hop_length = 128

# Diffusion params
guidance_scale = 3
n_candidates = 1
batch_size = 1
ddim_steps = 100


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
    audio = sample(latent_diffusion, foley_language_phrase, batch_size=1, ddim_steps=100, guidance_scale=3.0, \
             random_seed=random_seed, disable_tqdmoutput=False)

    return audio


'''
Dynamic Audio Converter
'''
def audio_post_processor(audio, freq_limit, hop_length, stft_channels, sample_rate, loudness):

    loudness_meter = pyln.Meter(sample_rate)
    wav_compressed_simple = compress_spectrogram_simple(audio, limit=freq_limit, hop_length=hop_length, stft_channels=stft_channels, \
                                                    sample_rate=sample_rate, loudness_meter=loudness_meter, loudness=loudness)

    wav_compressed_w_centroid = compress_spectrogram_with_centroid(audio, limit=freq_limit, hop_length=hop_length, stft_channels=stft_channels, \
                                                    sample_rate=sample_rate, loudness_meter=loudness_meter, loudness=loudness)

    _ = {\
         'original': audio, \
         'simple compressed': wav_compressed_simple, \
         'compressed with centroid': wav_compressed_w_centroid\
        }
    return _


def main(prompt):
    with open('config/.env', 'r') as fh:
        vars_dict = dict(
            tuple(line.replace('\n', '').split('='))
            for line in fh.readlines() if not line.startswith('#')
        )

    os.environ.update(vars_dict)

    latent_diffusion = get_model(model_name)
    client = OpenAI()

    foley_language_phrase = foley_interpreter(prompt, client)
    audio = audio_generator(foley_language_phrase, latent_diffusion, np.random.randint(0,10000))
    audio_c = audio_post_processor(audio, freq_limit=1000, hop_length=hop_length, stft_channels=stft_channels, sample_rate=sample_rate, loudness=-10)

    os.makedirs('output_dir', exist_ok=True)
    for k in audio_c:
        sf.write('output_dir/'+prompt.replace(' ','_')+'_'+k.replace(' ','_')+'.wav', data=audio_c[k], samplerate=sample_rate)

if __name__ == '__main__':

    prompt=sys.argv[1] 
    main(prompt)    