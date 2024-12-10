import os, json
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.pretrained import get_pretrained_model, load_local_trained_model

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.data.midi_utils import save_midi_and_wav
from stable_audio_tools.data.event_codec import Codec

import hook

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download model
# model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")

# model_config_path = "/home/intern-2024-04/SAO/stable_audio/stable_audio_tools/configs/model_configs/txt2audio/stable_audio_2_0.json"
# model, model_config = load_local_trained_model(model_config_path, 
#                                                "/home/intern-2024-04/SAO/stable_audio/harmonai_train/fm2kbxyx/checkpoints/epoch=482-step=2700.ckpt",
#                                                )
model_config_path = "/home/intern-2024-04/SAO/stable_audio/stable_audio_tools/configs/model_configs/txt2audio/stable_audio_3_0.json"
dataloader_config_path = "/home/intern-2024-04/SAO/stable_audio/stable_audio_tools/configs/dataset_configs/stage_4_datasets.json"
# model, model_config = load_local_trained_model(model_config_path, 
#                                                "/home/intern-2024-04/SAO/stable_audio/harmonai_train/yf1220yb/checkpoints/epoch=353-step=20600.ckpt",
#                                             #    "/home/intern-2024-04/SAO/stable_audio/harmonai_train/n9gmvk7i/checkpoints/epoch=66-step=3700.ckpt",
#                                                )

# model = model.to(device)
# hook()

with open(model_config_path) as f:
    model_config = json.load(f)

sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
    
# # use the first demo data as conditioning
# conditioning = [model_config['training']['demo']['demo_cond'][0]]
# print(conditioning[0]['prompt'])

# for batch, get conditions from the dataloader
with open(dataloader_config_path) as f:
    dataloader_config = json.load(f)


midi_length = dataloader_config["midi_length"]
resolution = 100
codec = Codec(int(midi_length * resolution + 1))

dataloader = create_dataloader_from_config(dataloader_config, batch_size=1, num_workers=3, sample_rate=sample_rate, sample_size=sample_size, audio_channels=2, 
                                           split='train'
                                           )

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

for i, data in enumerate(dataloader):
    audio, info = data  # Unpack data
hook()

for i, data in enumerate(dataloader):
    audio, info = data  # Unpack data
    info = info[0]  # Unpack info
    condition = {
        "prompt": info["prompt"],
        "seconds_start": info["seconds_start"],
        "seconds_total": info["seconds_total"],
        "midi": info["midi"]
    }
    
    output_folder = os.path.join("outputs_musicnet", str(i))
    os.makedirs(output_folder, exist_ok=True)
    
    # # Generate stereo audio
    # output = generate_diffusion_cond(
    #     model,
    #     steps=100,
    #     cfg_scale=7,
    #     conditioning=[condition],
    #     sample_size=sample_size,
    #     sigma_min=0.3,
    #     sigma_max=500,
    #     sampler_type="dpmpp-3m-sde",
    #     device=device
    # )
    
    # # Rearrange audio batch to a single sequence
    # output = rearrange(output, "b d n -> d (b n)")
    
    # # Peak normalize, clip, convert to int16, and save to file
    # output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    # audio_file = os.path.join(output_folder, "output.wav")
    # torchaudio.save(audio_file, output, sample_rate)
    
    # Save prompt to text file
    prompt_file = os.path.join(output_folder, "prompt.txt")
    with open(prompt_file, "w") as f:
        f.write(condition["prompt"])
    
    # save tokens to text file
    tokens_file = os.path.join(output_folder, "tokens.txt")
    with open(tokens_file, "w") as f:
        f.write(str(condition["midi"].tolist()))
        
    # save times to text file
    times_file = os.path.join(output_folder, "times.txt")
    with open(times_file, "w") as f:
        f.write(str(condition["seconds_start"]) + " " + str(condition["seconds_total"]))
        
    # save metadata to text file
    metadata_file = os.path.join(output_folder, "meta.txt")
    with open(metadata_file, "w") as f:
        f.write("audio_path: " + str(info['audio_path']) + "\n")
        f.write("file_idx: " + str(info['file_idx'])  + "\n")
        f.write("chunk_idx: " + str(info['chunk_idx']))
        
    
    # Save MIDI input and ground truth audio
    audio = audio[0]
    save_midi_and_wav("", "gt", info["midi"], codec, audio, sample_rate, output_folder=output_folder, synth_midi=True)

    
    print(f"Processed and saved output for example {i} at {output_folder}")

hook()

# Generate stereo audio
output = generate_diffusion_cond(
    model,
    steps=100,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device
)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("output.wav", output, sample_rate)
