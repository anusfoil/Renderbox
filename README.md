# RenderBox: Expressive performance rendering with text control




# Dependencies and requirements

To run the training scripts or inference code, you'll want to clone this repository, navigate to the root, and run:
```bash
$ pip install .
```
Install into a fresh environment, since several dependencies are version-sensitive and the install will move them. Python 3.10 with torch 2.4.1 (cu124) is the combination these instructions were verified against; install torch first if you need a specific CUDA build.

This project is developed upon codebase of [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools).

The controlnet experiment reported in the paper is adapted from [stable-audio-controlnet](https://github.com/EmilianPostolache/stable-audio-controlnet) and is kept in a separate repository, which is not part of this release.

# Inference 

Both entry points take the *unwrapped* checkpoint described under Weights below.

Launch the gradio page: ```python gradio_inference.py --unwarped_path=/path/to/checkpoint```. 

Full-midi inference: 

```python full_midi_by_segments.py --unwarped_path=/path/to/checkpoint --midi=/path/to/input --output=/path/to/output --prompt_file=/path/to/prompt/txt```
Note that this uses a very simple concatenation of segments with a bit cross-fading. 

Each segment is generated at the full ```sample_size``` of 47.5 seconds and trimmed afterwards, so decoding through the VAE dominates memory use: expect to need around 18GB of GPU memory, i.e. a 16GB card is not enough. The text conditioner pulls ```t5-base``` from HuggingFace on first run.


# Weights 
The full checkpoint, as described in the paper, is trained on datasets sourced from YouTube audio and thus cannot be shared publicly here. 
However, we trained a reduced version with only ```ASAP``` and ```MusicNet``` which are free-licensed audio datasets, up to stage 3, which can be accessed here ([RenderBox-open](https://drive.google.com/file/d/1unpf4eyYLfgFJPURCGQPRbaKJReFu9q3/view?usp=sharing)). Note that this checkpoint can do synthesis and performance rendering, but is not capable of performer or description conditioning. Nor is it able to generate instruments outside of these two datasets, such as guitar or saxophone. 

That file is the raw training checkpoint (17.5GB, it still carries the EMA and optimizer states). Unwrap it once before running inference:

```bash
$ python unwrap_model.py \
    --model-config stable_audio_tools/configs/model_configs/txt2audio/stable_audio_3_0.json \
    --ckpt-path /path/to/renderbox_open.ckpt \
    --name renderbox_open_unwrap
```

This writes ```renderbox_open_unwrap.ckpt``` (4.9GB), which is what ```--unwarped_path``` expects. Unwrapping restores the checkpoint to the devices it was saved on, so it wants a GPU big enough for the optimizer state as well; add ```--map-location cpu``` if yours is smaller. 

