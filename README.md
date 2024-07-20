# prompt-to-touch


## Work In Progress Readme.


Important: Create .env file under config folder with Open AI creds. 

### Setup
* Clone this repo
* Install dependencies (from the original AudioLDM repo as shown below) by creating a new conda environment called ```prompt-to-touch```
```
conda create -n prompt-to-touch python=3.10; conda activate prompt-to-touch
/home/purnima/anaconda3/envs/prompt-to-touch/bin/pip install git+https://github.com/haoheliu/AudioLDM2.git
/home/purnima/anaconda3/envs/prompt-to-touch/bin/pip install noisereduce  
/home/purnima/anaconda3/envs/prompt-to-touch/bin/pip install torch_pitch_shift  
/home/purnima/anaconda3/envs/prompt-to-touch/bin/pip install cython==0.29.19  
/home/purnima/anaconda3/envs/prompt-to-touch/bin/pip install tifresi==0.1.2  
/home/purnima/anaconda3/envs/prompt-to-touch/bin/pip install pyloudnorm
/home/purnima/anaconda3/envs/prompt-to-touch/bin/pip install openai
/home/purnima/anaconda3/envs/prompt-to-touch/bin/pip pip install ipython
pip install audio_dspy
```
audio_dspy: https://audio-dspy.readthedocs.io/en/latest/index.html (make note)
  
Add the newly created environment to Jupyter Notebooks
```
python -m ipykernel install --user --name prompt-to-touch
```


conda create -n prompt-to-touch python=3.10; conda activate prompt-to-touch

