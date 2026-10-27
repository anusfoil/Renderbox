from setuptools import setup, find_packages

setup(
    name='stable-audio-tools',
    version='0.0.16',
    url='https://github.com/Stability-AI/stable-audio-tools.git',
    author='Stability AI',
    description='Training and inference tools for generative audio models from Stability AI',
    packages=find_packages(),  
    install_requires=[
        # 0.0.20 pins soundfile<=0.10.2, which needs a system libsndfile and
        # conflicts with the wheel-bundled soundfile below. 0.0.21 drops the pin.
        'aeiou>=0.0.21',
        'alias-free-torch==0.0.6',
        'auraloss==0.4.0',
        'descript-audio-codec==1.0.0',
        'einops==0.7.0',
        'einops-exts==0.0.4',
        # The released checkpoints store the EMA `initted`/`step` buffers as scalars,
        # which 0.2.x writes as shape [1]; unwrapping fails on that mismatch.
        'ema-pytorch>=0.4',
        'encodec==0.1.1',
        'gradio>=3.42.0',
        'huggingface_hub==0.24.6',
        'importlib-resources==5.12.0',
        'k-diffusion==0.1.1',
        'laion-clap==1.1.4',
        'local-attention==1.8.6',
        'pandas==2.0.2',
        'pedalboard==0.7.4',
        'prefigure==0.0.9',
        'pytorch_lightning==2.1.0',
        'PyWavelets==1.4.1',
        'safetensors',
        'sentencepiece==0.1.99',
        # clip, imported by k-diffusion, still does `from pkg_resources import packaging`,
        # which setuptools dropped in 67.
        'setuptools<66',
        's3fs',
        'torch>=2.0.1',
        'torchaudio>=2.0.2',
        'torchmetrics==0.11.4',
        'tqdm',
        # 5.x moves the T5 encoder APIs the conditioner uses.
        'transformers==4.44.2',
        'v-diffusion-pytorch==0.0.2',
        'vector-quantize-pytorch==1.9.14',
        # 0.15.4 cannot import against the protobuf that note_seq pulls in.
        'wandb>=0.17',
        'webdataset==0.2.48',
        'x-transformers<1.27.0',

        # Required by the RenderBox MIDI conditioning and alignment code, which
        # upstream stable-audio-tools does not have.
        'fastdtw',
        'mido',
        'note_seq',
        'parangonar',
        'partitura',
        'pretty_midi',
        'soundfile>=0.12.1',
    ],
)