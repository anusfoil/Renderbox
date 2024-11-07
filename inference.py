import torch
import torchaudio
from einops import rearrange
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.pretrained import get_pretrained_model, load_local_trained_model

import hook

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download model
# model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model, model_config = load_local_trained_model("/home/intern-2024-04/SAO/stable_audio/stable_audio_tools/configs/model_configs/txt2audio/stable_audio_2_0.json", 
                                               "/home/intern-2024-04/SAO/stable_audio/harmonai_train/tj57otg2/checkpoints/epoch=128-step=900.ckpt",
                                               )
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

# Set up text and timing conditioning
conditioning = [{
    "prompt": "Bach, Prelude, bwv_846, expressive performance",
    "seconds_start": 0, 
    "seconds_total": 47
}]

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
