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
    content = "Describe a language phrase consisting of a noun verb adjective adverb like "+\
            "'a brown cat running purposefully and quickly' that would be a sound which could be also perceivable "+\
            "as touch sensation and resembles "+txt+". Use simple words and long descriptions when generating the language phrase."
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
         {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content}
      ]
    )

    foley_language_phrase = response.choices[0].message.content.split('"')[1]
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