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
from stable_audio_tools.inference.postprocess import normalize, render_until_valid

import hook

device = "cuda" if torch.cuda.is_available() else "cpu"
stage = "0"


model_config_path = "stable_audio_tools/configs/model_configs/txt2audio/stable_audio_3_0.json"
dataloader_config_path = f"stable_audio_tools/configs/dataset_configs/atepp.json"
# dataloader_config_path = f"stable_audio_tools/configs/dataset_configs/stage_{stage}_datasets.json"
model, model_config = load_local_trained_model(model_config_path, 
                                            #    "harmonai_train/xr4laxjt/checkpoints/epoch=800-step=61500.ckpt",
                                               "harmonai_train/lpt77c2n/checkpoints/epoch=620-step=43200.ckpt",
                                            #    unwarped_path="model_unwrap.ckpt"
                                            #    "harmonai_train/qjafgbey/checkpoints/epoch=735-step=47600.ckpt",
                                            #    "harmonai_train/n9gmvk7i/checkpoints/epoch=347-step=19100.ckpt",
                                               )

model = model.to(device)
hook()

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


PERFORMERS = ['Sviatoslav Richter', 'Vladimir Horowitz', 'Vladimir Ashkenazy',
       'Grigory Sokolov', 'Hélène Grimaud', 'Nikolai Lugansky',
       'Valentina Lisitsa', 'Yuja Wang', 'Zoltán Kocsis', 'Lazar Berman',
       'Emil Gilels', 'Evgeny Kissin', 'Leif Ove Andsnes',
       'Daniel Barenboim', 'Alfred Brendel', 'Mikhail Pletnev',
       'Claudio Arrau', 'Alfred Cortot', 'András Schiff',
       'Martha Argerich', 'Dinu Lipatti', 'Alice Sara Ott',
       'Bertrand Chamayou', 'Anne Queffélec', 'Lang Lang',
       'Walter Gieseking', 'Murray Perahia', 'Alicia de Larrocha',
       'Radu Lupu', 'Wilhelm Kempff', 'Mitsuko Uchida',
       'Maurizio Pollini', 'Glenn Gould', 'Friedrich Gulda',
       'Edwin Fischer', 'Ivo Pogorelich', 'Arturo Benedetti Michelangeli',
       'Robert Casadesus', 'Pierre-Laurent Aimard', 'Krystian Zimerman',
       'Samson François', 'Alexandre Tharaud', 'Artur Schnabel',
       'Myra Hess', 'Emanuel Ax', 'Van Cliburn']

for i, data in enumerate(dataloader):
    
    if i < 110 or i > 500:
        continue
    
    audio, info = data  # Unpack data
    audio = audio[0]
    info = info[0]  # Unpack info

    performers = ["Glenn Gould", "Martha Argerich", "Vladimir Horowitz", "Maurizio Pollini", "Sviatoslav Richter", "Yuja Wang", "Emil Gilels", "Alfred Cortot", "Evgeny Kissin", "Van Cliburn"]

    for performer in performers:

        output_folder = f"outputs/demo_latent/{str(i)}/{performer.split(' ')[-1]}"
        os.makedirs(output_folder, exist_ok=True)
        
        prompt = f"In the style of {performer}, expressive"
        
        condition = {
            "prompt": prompt,
            "seconds_start": info["seconds_start"],
            "seconds_total": 30,
            "midi": info["midi"]
        }

        
        # Generate stereo audio. The model always fills the full 47.55s canvas
        # and only the first seconds_total carry music, so the rest has to be
        # trimmed; a draw that collapsed to the silence mode is redrawn with a
        # new seed rather than written out.
        def draw(attempt, condition=condition, output_folder=output_folder):
            out = generate_diffusion_cond(
                model,
                steps=100,
                cfg_scale=6,
                conditioning=[condition],
                sample_size=sample_size,
                sigma_min=0.03,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device=device,
                save_path=output_folder,
                seed=13137013 + attempt,
            )
            return rearrange(out, "b d n -> d (b n)").to(torch.float32)

        output, diag = render_until_valid(
            draw, sample_rate, condition["seconds_total"],
            on_reject=lambda a, d: print(f"  rejected draw {a + 1}: {d['reason']}"))
        if output is None:
            print(f"  SKIPPED example {i} / {performer}: {diag['reason']}")
            continue

        output = normalize(output)
        audio_file = os.path.join(output_folder, "output.wav")
        torchaudio.save(audio_file, output.mul(32767).to(torch.int16).cpu(), sample_rate)
        
        # Save prompt to text file
        prompt_file = os.path.join(output_folder, "prompt.txt")
        with open(prompt_file, "w") as f:
            f.write(prompt)
        
        # save tokens to text file
        tokens_file = os.path.join(output_folder, "tokens.txt")
        with open(tokens_file, "w") as f:
            f.write(str(condition["midi"].tolist()))
            
        # # save times to text file
        # times_file = os.path.join(output_folder, "times.txt")
        # with open(times_file, "w") as f:
        #     f.write(str(condition["seconds_start"]) + " " + str(condition["seconds_total"]))
            
        # save metadata to text file
        metadata_file = os.path.join(output_folder, "meta.txt")
        with open(metadata_file, "w") as f:
            f.write("audio_path: " + str(info['audio_path']) + "\n")
            f.write("file_idx: " + str(info['file_idx'])  + "\n")
            f.write("chunk_idx: " + str(info['chunk_idx']))
            
        
        # Save MIDI input and ground truth audio
        # save_midi_and_wav("", "gt", info["midi"], codec, audio, sample_rate, output_folder=output_folder, synth_midi=True)


        print(f"Processed and saved output for example {i} at {output_folder}")

hook()
