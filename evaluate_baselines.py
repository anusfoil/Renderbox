import os, json, parser
from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.data.midi_utils import save_midi_and_wav
from stable_audio_tools.data.event_codec import Codec

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import hook
import torch
import torchaudio
from einops import rearrange
device = "cuda" if torch.cuda.is_available() else "cpu"





def import_by_envs(args):
    if args.baseline == "midi_ddsp":
        print("don't forget to use the environment 'mdenv'! ")
        from midi_ddsp import synthesize_midi, load_pretrained_model
        return synthesize_midi, load_pretrained_model
    elif args.baseline == "coco_mulla":
        print("don't forget to use the environment 'cmenv'! ")
        pass
    elif args.baseline == "musicgen_melody":
        print("don't forget to use the environment 'mgenv'! ")
        from audiocraft.models import MusicGen
        from audiocraft.data.audio import audio_write
        return MusicGen, audio_write


def main(args, modules):

    dataloader_config_path = f"/home/intern-2024-04/SAO/stable_audio/stable_audio_tools/configs/dataset_configs/stage_{args.stage}_datasets.json"


    # for batch, get conditions from the dataloader
    with open(dataloader_config_path) as f:
        dataloader_config = json.load(f)

    sample_rate = 44100
    sample_size = 2097152

    midi_length = dataloader_config["midi_length"]
    resolution = 100
    codec = Codec(int(midi_length * resolution + 1))

    dataloader = create_dataloader_from_config(dataloader_config, batch_size=1, num_workers=3, sample_rate=sample_rate, sample_size=sample_size, audio_channels=2, 
                                            split='train'
                                            )

    if args.baseline == "musicgen_melody":
        MusicGen, audio_write = modules
        
        model = MusicGen.get_pretrained('melody')
        
    elif args.baseline == "stable-audio-open":
        from stable_audio_tools import get_pretrained_model
        from stable_audio_tools.inference.generation import generate_diffusion_cond

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Download model
        model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]

        model = model.to(device)


    for i, data in enumerate(dataloader):
        audio, info = data  # Unpack data
        info = info[0]  # Unpack info
        condition = {
            "prompt": info["prompt"],
            "seconds_start": info["seconds_start"],
            "seconds_total": info["seconds_total"],
            "midi": info["midi"]
        }
        
        output_folder = os.path.join(f"outputs/stage_{args.stage}_{args.baseline}", str(i))
        os.makedirs(output_folder, exist_ok=True)

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


        # Save MIDI input and ground truth audio
        audio = audio[0]
        save_midi_and_wav("", "gt", info["midi"], codec, audio, sample_rate, output_folder=output_folder, synth_midi=True)

            
        # save metadata to text file
        metadata_file = os.path.join(output_folder, "meta.txt")
        with open(metadata_file, "w") as f:
            f.write("audio_path: " + str(info['audio_path']) + "\n")
            f.write("file_idx: " + str(info['file_idx'])  + "\n")
            f.write("chunk_idx: " + str(info['chunk_idx']))
        
        if args.baseline == "midi_ddsp":
            synthesize_midi, load_pretrained_model = modules
            
            midi_file = os.path.join(output_folder, 'gt.mid')
            synthesis_generator, expression_generator = load_pretrained_model()
            output = synthesize_midi(synthesis_generator, expression_generator, 
                                     midi_file, speed_rate=(10 / condition["seconds_total"]),
                                     use_fluidsynth=True, sf2_path='/work/GeneralUser-GS.sf2',
                                    output_dir = output_folder, save_metadata=True)  # it will be called 'mix.wav' in the output folder
            
            
        elif args.baseline == "coco_mulla":
            os.system(f"CUDA_VISIBLE_DEVICES=1 python /home/intern-2024-04/coco-mulla-repo/inference.py --num_layers=48 --latent_dim=12 \
                    --output_folder={output_folder} \
                    --model_path=/home/intern-2024-04/coco-mulla-repo/diff_9_end.pth \
                    --midi_path={os.path.join(output_folder, 'gt.mid')} \
                    --prompt_path={prompt_file} \
                    --offset=0  ")
        
        
        elif args.baseline == "musicgen_melody":

            model.set_generation_params(duration=info['seconds_total']) 
            descriptions = info["prompt"].split(", ")

            melody_file = os.path.join(output_folder, 'gt.wav')
            melody, sr = torchaudio.load(melody_file)
            wav = model.generate_with_chroma(descriptions, melody[None], sr)

            # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
            save_path = os.path.join(output_folder, f'output')
            audio_write(save_path, wav[0].cpu(), model.sample_rate, strategy="loudness")

        elif args.baseline == "stable-audio-open":

            # Set up text and timing conditioning
            conditioning = [{
                "prompt": info["prompt"].split(", "),
                "seconds_start": 0, 
                "seconds_total": info["seconds_total"],
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
            torchaudio.save(os.path.join(output_folder, f'output.wav'), output, sample_rate)


        
        print(f"Processed and saved output for example {i} at {output_folder}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate baselines for evaluation')
    parser.add_argument('--baseline', type=str, help='Name of pretrained model')
    parser.add_argument('--stage', type=int, default=0, help='stage of test data to infer on, can be 0, 1, 2, 3')

    args = parser.parse_args()
    
    modules = import_by_envs(args)
    main(args, modules)