import os, subprocess
import json
import torch
import torchaudio
import gradio as gr
from einops import rearrange
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.pretrained import load_local_trained_model
from stable_audio_tools.data.midi_utils import tokens_to_notes, note_sequence_to_midi_file
from stable_audio_tools.data.event_codec import Codec
from stable_audio_tools.data.dataset import midi_to_tokens
import hook

# Model and device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model_config_path = "/home/intern-2024-04/SAO/stable_audio/stable_audio_tools/configs/model_configs/txt2audio/stable_audio_3_0.json"
checkpoint_path = "/home/intern-2024-04/SAO/stable_audio/harmonai_train/6kn4j206/checkpoints/epoch=350-step=16700.ckpt"
model, model_config = load_local_trained_model(model_config_path, checkpoint_path)
with open(model_config_path) as f:
    model_config = json.load(f)
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
model = model.to(device)

# Codec for MIDI
midi_length = 10
resolution = 100
codec = Codec(int(midi_length * resolution + 1))


# Load presets from JSON file
def load_presets(presets_file="gradio_presets.json"):
    with open(presets_file, "r") as f:
        return json.load(f)

presets = load_presets()


# Function for audio generation
def generate_audio(prompt, seconds_total, midi_tokens, output_folder="gradio_outputs"):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    if isinstance(midi_tokens, str):
        midi_tokens = torch.tensor(eval(midi_tokens))

    # Create condition
    condition = {
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": seconds_total,
        "midi": midi_tokens
    }

    # Generate audio
    output = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=7,
        conditioning=[condition],
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )

    # Process and save audio
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    audio_file = os.path.join(output_folder, "output.wav")
    torchaudio.save(audio_file, output, sample_rate)

    return audio_file


# Gradio interface
def gradio_generate(prompt, seconds_total, midi_tokens, preset=None):
    if preset and preset in presets:
        preset_data = presets[preset]
        prompt = preset_data["prompt"]
        seconds_total = preset_data["seconds_total"]
        midi_tokens = preset_data["midi_tokens"]
    
    output_folder = os.path.join("outputs", "gradio")
    audio_file = generate_audio(prompt, seconds_total, midi_tokens, output_folder)
    
    note_sequence = tokens_to_notes(torch.tensor(eval(midi_tokens)), codec)
    midi_path = os.path.join(output_folder, f'input.mid')
    note_sequence_to_midi_file(note_sequence, midi_path)
    
    midi_synthesized = os.path.join(output_folder, f'midi_synthesized.wav')
    try:
        subprocess.run(
            ["fluidsynth", "-ni", "GeneralUser-GS.sf2", midi_path, "-F", midi_synthesized],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"FluidSynth error: {e.stderr.decode()}")
        raise e
    
    return audio_file, midi_synthesized

# Auto-populate fields on preset selection
def auto_populate(preset):
    if preset and preset in presets:
        preset_data = presets[preset]
        return preset_data["prompt"], json.dumps(preset_data["midi_tokens"], indent=2), preset_data["seconds_total"]
    return "", "", 10

# Function to tokenize MIDI file (you'll provide the actual implementation)
def tokenize_midi_file(midi_file_path):
    midi_tokens = midi_to_tokens(midi_file_path, codec, midi_length=midi_length)
    
    return midi_tokens[0]

# Function to handle MIDI file upload and tokenization
def upload_and_tokenize_midi(midi_file):
    if midi_file is None:
        return "No file uploaded"
    
    # Save the uploaded MIDI file
    file_path = midi_file.name
    
    # Tokenize the MIDI file
    tokens = tokenize_midi_file(file_path)
    
    # Convert tokens to JSON format for display in the textbox
    tokens_json = json.dumps(tokens.tolist(), indent=2)
    return tokens_json

# Build the Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("# RenderBox: Expressive Performance Rendering")
    gr.Markdown("This interface generates expressive audio from a text prompt and MIDI tokens (from note_seq format). You can choose from the presets and it will fill in the text prompt and MIDI tokens from our dataset. \
        If uploading a MIDI file, please click the 'Tokenize MIDI' button and it will overwrite the text in the MIDI tokens box. The window of MIDI input is 10 seconds, and it will only take the first 10 seconds of the given MIDI. \
        The duration (in seconds) is the desired output audio length (maximum 47), and it should also be related to the speed keywords given in the prompt (i.e. 'about half the speed of' ~ 20 seconds).")
    
    with gr.Row():
        preset_selector = gr.Dropdown(
            choices=list(presets.keys()),
            label="Presets (Optional)",
            interactive=True
        )

    with gr.Row():
        prompt_input = gr.Textbox(
            label="Text Prompt",
            placeholder="Enter a description of the music",
            lines=4,
            interactive=True
        )
        midi_input = gr.Textbox(
            label="MIDI Tokens",
            placeholder="Enter MIDI tokens as a list of tuples (pitch, start_time, duration, velocity)",
            lines=8,
            interactive=True
        )

    with gr.Row():
        midi_file_upload = gr.File(
            label="Upload MIDI File",
            file_types=[".mid", ".midi"],
            interactive=True
        )
        tokenize_button = gr.Button("Tokenize MIDI")

    seconds_input = gr.Number(label="Duration (seconds)", value=10, interactive=True)

    generate_button = gr.Button("Generate")
    audio_output = gr.Audio(label="Generated Audio", type="filepath")
    midi_input_synthesized = gr.Audio(label="MIDI Input", type="filepath")

    # Auto-populate on preset selection
    preset_selector.change(
        auto_populate,
        inputs=[preset_selector],
        outputs=[prompt_input, midi_input, seconds_input]
    )

    # Update MIDI tokens when a MIDI file is uploaded and tokenized
    tokenize_button.click(
        upload_and_tokenize_midi,
        inputs=[midi_file_upload],
        outputs=[midi_input]
    )

    # Generate audio on button click
    generate_button.click(
        gradio_generate,
        inputs=[prompt_input, seconds_input, midi_input],
        outputs=[audio_output, midi_input_synthesized]
    )


# Launch the interface
if __name__ == "__main__":
    interface.launch()
