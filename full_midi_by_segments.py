import os, json
import numpy as np
import torch
import torchaudio
import pretty_midi
from einops import rearrange
import hook

from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.pretrained import get_pretrained_model, load_local_trained_model

from stable_audio_tools.data.dataset import midi_to_tokens
from stable_audio_tools.data.midi_utils import save_midi_and_wav
from stable_audio_tools.data.event_codec import Codec


device = "cuda" if torch.cuda.is_available() else "cpu"
model_config_path = "stable_audio_tools/configs/model_configs/txt2audio/stable_audio_3_0.json"

with open(model_config_path) as f:
    model_config = json.load(f)

sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]



def remove_silence(audio, threshold=0.01):
    """Remove silence from the beginning and end of an audio tensor."""
    energy = audio.pow(2).mean(dim=0)
    non_silent_indices = (energy > threshold).nonzero(as_tuple=True)[0]
    if len(non_silent_indices) == 0:
        return audio
    start, end = non_silent_indices[0], non_silent_indices[-1]
    return audio[:, start:end]

def crossfade_audio(audio1, audio2, fade_duration, sample_rate):

    fade_samples = int(fade_duration * sample_rate)
    fade_out = torch.linspace(1, 0, fade_samples).to(audio1.device)
    fade_in = torch.linspace(0, 1, fade_samples).to(audio1.device)
    fade_samples_ = min(fade_samples, audio1.size(1), audio2.size(1))
    crossfaded = audio1[:, -fade_samples_:] * fade_out + audio2[:, :fade_samples_] * fade_in
    return torch.cat([audio1[:, :-fade_samples], crossfaded, audio2[:, fade_samples:]], dim=1)


def process_midi(midi_file, model, codec, sample_rate, output_file, midi_length, prompt):
    """Process a MIDI file segment by segment."""

    final_audio = None
    midi_tokens = midi_to_tokens(midi_file, codec, midi_length=midi_length)

    for i, token in enumerate(midi_tokens):

        midi_tokens = midi_to_tokens(midi_file, codec, midi_length=midi_length)
        
        condition = {
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": 10,  # Note: set this yourself!!! 
            "midi": token
        }
        
        if i == len(midi_tokens) - 1:
            condition["seconds_total"] = 17  # last segment length; set it yourself!!!

        output = generate_diffusion_cond(
            model,
            steps=100,
            cfg_scale=9,
            conditioning=[condition],
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=device
        )

        output = rearrange(output, "b d n -> d (b n)")
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)

        output = remove_silence(output)

        # Concatenate segments with crossfading
        if final_audio is None:
            final_audio = output
        else:
            final_audio = crossfade_audio(final_audio, output, fade_duration=0.05, sample_rate=sample_rate)

    final_audio = final_audio.mul(32767).to(torch.int16).cpu()
    torchaudio.save(output_file, final_audio, sample_rate)



if __name__ == "__main__":

    midi_length = 10
    resolution = 100
    codec = Codec(int(midi_length * resolution + 1))
        
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unwarped_path", type=str, required=True)
    parser.add_argument("--midi", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    args = parser.parse_args()

    prompt = open(args.prompt_file, "r").read()

    model, _ = load_local_trained_model(model_config_path, "", unwarped_path=args.unwarped_path)
    model = model.to(device)

    process_midi(args.midi, model, codec, sample_rate, args.output, midi_length, prompt)




