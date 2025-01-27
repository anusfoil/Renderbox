# RenderBox: Expressive performance rendering with text control




# Dependencies and requirements

To run the training scripts or inference code, you'll want to clone this repository, navigate to the root, and run:
```bash
$ pip install .
```
This project is developed upon codebase of [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools).

The controlnet experiment is documented in the ```main_controlnet/``` directory, which is adapated from [stable-audio-controlnet](https://github.com/EmilianPostolache/stable-audio-controlnet).

# Inference 

Launch the gradio page: ```python gradio_inference.py --unwarped_path=/path/to/checkpoint```. 

Full-midi inference: 

```python full_midi_by_segments.py --unwarped_path=/path/to/checkpoint --midi=/path/to/input --output=/path/to/output --prompt_file=/path/to/prompt/txt```
Note that this uses a very simple concatenation of segments with a bit cross-fading. 


# Weights 
The full checkpoint, as described in the paper, is trained on datasets sourced from YouTube audio and thus cannot be shared to general public here. 
However, we trained a reducted version with only ```ASAP``` and ```MusicNet``` which are free-licensed audio datasets, up to stage 3, which can be accessed here ([RenderBox-open]()). Note that this checkpoint can do synthesis and performance rendering, but is not capable of performer or description conditioning. Nor it's able to generate instruments outside of these two datasets, such as guitar or saxophone. 

