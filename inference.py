import os, json, random
import torch
import torchaudio
import pretty_midi
import matplotlib.pyplot as plt
from einops import rearrange
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.pretrained import get_pretrained_model, load_local_trained_model

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.data.midi_utils import save_midi_and_wav
from stable_audio_tools.data.event_codec import Codec

import hook

device = "cuda" if torch.cuda.is_available() else "cpu"
stage = "0"


model_config_path = "/home/intern-2024-04/SAO/stable_audio/stable_audio_tools/configs/model_configs/txt2audio/stable_audio_3_0.json"
dataloader_config_path = f"/home/intern-2024-04/SAO/stable_audio/stable_audio_tools/configs/dataset_configs/stage_2_datasets.json"
# dataloader_config_path = f"/home/intern-2024-04/SAO/stable_audio/stable_audio_tools/configs/dataset_configs/stage_{stage}_datasets.json"
model, model_config = load_local_trained_model(model_config_path, 
                                               "/home/intern-2024-04/SAO/stable_audio/harmonai_train/xr4laxjt/checkpoints/epoch=786-step=60500.ckpt",
                                               unwarped_path="/home/intern-2024-04/SAO/stable_audio/model_unwrap.ckpt"
                                            #    "/home/intern-2024-04/SAO/stable_audio/harmonai_train/qjafgbey/checkpoints/epoch=735-step=47600.ckpt",
                                            #    "/home/intern-2024-04/SAO/stable_audio/harmonai_train/n9gmvk7i/checkpoints/epoch=347-step=19100.ckpt",
                                               )

model = model.to(device)
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

# seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
dataloader = create_dataloader_from_config(dataloader_config, batch_size=1, num_workers=1, sample_rate=sample_rate, sample_size=sample_size, audio_channels=2, 
                                           split='test'
                                           )

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# for i, data in enumerate(dataloader):
#     audio, info = data  # Unpack data
# hook()


DESCRIPTIONS = [
    ("slow, sad, clear phrasing", "fast, energetic, dynamic"),
    ("soft, tender, delicate, calm", "hard, loud, aggressive"),
    ("dreamy, flowing, serene", "mechanical, staccato, rigid"),
    ("slow, melancholic, with gravity", "brisk, lively, light-hearted"),
    ("delicate, expressive, subtle", "bold, dramatic, commanding"),
    ("steady, deliberate, introspective", "hurried, rushed, chaotic"),
    ("romantic, expressive, lush", "flat, emotionless, robotic"),
    ("flowing, calm, well-phrased", "jerky, uneven, unpredictable"),
    ("slow, introspective, thoughtful", "fast, agitated, restless"),
    ("soft, airy, ethereal", "powerful, dramatic, intense")
]

for i, data in enumerate(dataloader):
    if i > 5:
        continue
    
    audio, info = data  # Unpack data
    audio = audio[0]
    info = info[0]  # Unpack info


    # Define output folder for this data point
    base_output_folder = f"outputs/demo_comparisons/{str(i)}"
    os.makedirs(base_output_folder, exist_ok=True)

    # 0: Reference MIDI audio itself
    save_midi_and_wav(
        "", "input", info["midi"], codec, audio, sample_rate, output_folder=base_output_folder, synth_midi=True
    )

    import numpy as np

    # Save piano roll for synthesized input, loaded from the MIDI file
    midi_file = os.path.join(base_output_folder, "input.mid")
    piano_roll_file = os.path.join(base_output_folder, "piano_roll_input.png")

    # Load MIDI file using pretty_midi
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    # Get the piano roll
    fs = 100  # Frames per second
    piano_roll = midi_data.get_piano_roll(fs=fs)  # Resolution in frames per second

    # Pad or truncate the piano roll to 20 seconds
    target_frames = 20 * fs
    if piano_roll.shape[1] < target_frames:
        # Pad with zeros if the piano roll is shorter than 20 seconds
        piano_roll = np.pad(piano_roll, ((0, 0), (0, target_frames - piano_roll.shape[1])), mode="constant")
    else:
        # Truncate if the piano roll is longer than 20 seconds
        piano_roll = piano_roll[:, :target_frames]

    # Plot the piano roll without labels or extras
    plt.figure(figsize=(10, 4))
    plt.imshow(
        piano_roll,
        aspect="auto",
        origin="lower",
        cmap="Reds",
    )
    # Add a border frame
    ax = plt.gca()  # Get current axes
    for spine in ax.spines.values():
        spine.set_edgecolor("red")  # Set border color
        spine.set_linewidth(2)        # Set border thickness

    plt.axis('off')  # Remove all axis labels and ticks
    plt.tight_layout()
    plt.savefig(piano_roll_file, bbox_inches='tight', pad_inches=0)
    plt.close()

    
    # Save spectrogram for synthesized input, needs to be loaded from input_synthmid.wav
    spec_file = os.path.join(base_output_folder, "spectrogram_input.png")
    spec_data = torchaudio.load(os.path.join(base_output_folder, "input_synthmid.wav"))[0]
    spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=256)(spec_data.float())
    spec = torch.clamp(spec, min=1e-5)  # Replace zeros with a small value
    log_spec = spec.log2()
    plt.figure(figsize=(10, 4))
    plt.imshow(
        log_spec[0, :, :].numpy(),
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=-50,  # Minimum dB for the colormap
        vmax=10,   # Maximum dB for the colormap
    )
    plt.axis('off') # Hide axes
    plt.tight_layout()
    plt.savefig(spec_file)
    plt.close()


    # Define comparison prompts
    description = random.choice(DESCRIPTIONS)
    comparisons = [
        ("direct_synthesis", f"direct synthesis"),
        ("speed_change", random.choice(["synthesis, a bit faster", "synthesis, a bit slower"])),  
        ("mistakes", "performance of a student, with error"),
    ] + [
        (f"performance_{j}", desc) for j, desc in enumerate(description)
    ]

    for label, description in comparisons:
        output_folder = os.path.join(base_output_folder, label)
        os.makedirs(output_folder, exist_ok=True)
        
        if "performance" in label:
            if "asap" in info["audio_path"]:
                prompt = f"performance by {random.choice(['Glenn Gould', 'Grigory Sokolov'])}"
            else:
                prompt = f"expressive performance, {description}"
            
            prompt = prompt + random.choice(["", ", a bit faster", ", a bit slower", ", considerably faster", ", considerably slower"])
        else:
            prompt = description

        prompt_file = os.path.join(output_folder, "prompt.txt")
        with open(prompt_file, "w") as f:
            f.write(prompt)
        
        condition = {
            "prompt": prompt,
            "seconds_start": info["seconds_start"],
            "seconds_total": 30,
            "midi": info["midi"]
        }

        # Generate output
        output = generate_diffusion_cond(
            model,
            steps=100,
            cfg_scale=7,
            conditioning=[condition],
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=device,
        )
        
        # Rearrange audio and normalize
        output = rearrange(output, "b d n -> d (b n)")
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        
        # Save generated audio
        audio_file = os.path.join(output_folder, "output.wav")
        torchaudio.save(audio_file, output, sample_rate)

        # Preprocess output tensor
        output = output.to(torch.float32)  # Ensure float32 type
        output = output / torch.max(torch.abs(output))  # Peak normalize to [-1, 1]

        # Take the first 20 seconds
        spec_data = output[:, :20 * sample_rate]

        import torchaudio.transforms as T
        import librosa
        import librosa.display

        spec_file = os.path.join(output_folder, "spectrogram.png")
        spec_data_np = spec_data.numpy()
        # Assume `spec_data_np` is your raw audio numpy array
        spec = librosa.feature.melspectrogram(
            y=spec_data_np,
            sr=sample_rate,
            n_fft=2048,
            hop_length=128,
            n_mels=128,
        )
        log_spec = librosa.power_to_db(spec, ref=np.max)

        # Ensure spectrogram has correct shape
        log_spec = log_spec.squeeze()  # Remove singleton dimensions if any
        if log_spec.ndim == 3:
            log_spec = log_spec.mean(axis=0)  # Combine channels if stereo

        # Plot the spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            log_spec,
            sr=sample_rate,
            hop_length=128,
            x_axis=None,
            y_axis=None,
            cmap="inferno",  # High-contrast colormap
        )
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(spec_file, bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close()

        # Save metadata
        metadata_file = os.path.join(output_folder, "meta.txt")
        with open(metadata_file, "w") as f:
            f.write(f"audio_path: {info['audio_path']}\n")
            f.write(f"file_idx: {info['file_idx']}\n")
            f.write(f"chunk_idx: {info['chunk_idx']}\n")
            f.write(f"prompt: {prompt}\n")
        
        print(f"Processed and saved {label} for example {i} at {output_folder}")

    from PIL import Image

    # Define paths for piano roll and spectrograms
    base_output_folder = f"outputs/demo_comparisons/{i}"
    subfolders = sorted([f for f in os.listdir(base_output_folder) if os.path.isdir(os.path.join(base_output_folder, f))])
    piano_roll_file = os.path.join(base_output_folder, "piano_roll_input.png")
    images = [Image.open(piano_roll_file)]

    # Collect spectrogram files from subfolders
    for subfolder in subfolders:
        spectrogram_file = os.path.join(base_output_folder, subfolder, "spectrogram.png")
        if os.path.exists(spectrogram_file):
            images.append(Image.open(spectrogram_file))

    # Compute total width and height
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)

    # Create a new blank image
    stitched_image = Image.new("RGB", (total_width, max_height), "black")

    # Paste images side-by-side
    x_offset = 0
    for img in images:
        stitched_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    # Save the stitched image
    stitched_file = os.path.join(base_output_folder, "stitched_output.png")
    stitched_image.save(stitched_file)
    print(f"Stitched image saved at {stitched_file}")


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
