import importlib
import numpy as np
import io
import os
import posixpath
import random
import re
import subprocess
from itertools import chain
import time
import torch
import torchaudio
import webdataset as wds
import note_seq
import pandas as pd

from aeiou.core import is_silence
from os import path
from pedalboard.io import AudioFile
from torchaudio import transforms as T
from typing import Optional, Callable, List

from .utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T
from .midi_utils import preprocess, save_midi_and_wav
from .event_codec import Codec
from .alignment_utils import asap_alignment_segpoint, get_yamahasax_segpoints, get_musicnet_segpoints, gaps_alignment_segpoint, get_vienna422_segpoints, get_bachviolin_segpoints

AUDIO_KEYS = ("flac", "wav", "mp3", "m4a", "ogg", "opus")

# fast_scandir implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py

def fast_scandir(
    dir:str,  # top-level directory at which to begin scanning
    ext:list,  # list of allowed file extensions,
    #max_size = 1 * 1000 * 1000 * 1000 # Only files < 1 GB
    ):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    ext = ['.'+x if x[0]!='.' else x for x in ext]  # add starting period to extensions if needed
    try: # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try: # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")

                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except:
                pass 
    except:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def keyword_scandir(
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions
    keywords: list,  # list of keywords to search for in the file name
):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    # make keywords case insensitive
    keywords = [keyword.lower() for keyword in keywords]
    # add starting period to extensions if needed
    ext = ['.'+x if x[0] != '.' else x for x in ext]
    banned_words = ["paxheader", "__macosx"]
    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = f.name.split("/")[-1][0] == '.'
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    name_lower = f.name.lower()
                    has_keyword = any(
                        [keyword in name_lower for keyword in keywords])
                    has_banned = any(
                        [banned_word in name_lower for banned_word in banned_words])
                    if has_ext and has_keyword and not has_banned and not is_hidden and not os.path.basename(f.path).startswith("._"):
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for dir in list(subfolders):
        sf, f = keyword_scandir(dir, ext, keywords)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def get_audio_filenames(
    paths: list,  # directories in which to search
    keywords=None,
    exts=['.wav', '.mp3', '.flac', '.ogg', '.aif', '.opus']
):
    "recursively get a list of audio filenames"
    filenames = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:               # get a list of relevant filenames
        if keywords is not None:
            subfolders, files = keyword_scandir(path, exts, keywords)
        else:
            subfolders, files = fast_scandir(path, exts)
        filenames.extend(files)
    return filenames

def get_metadata_info(dataset_id, base_path, metadata_row, codec, midi_len=10):
        
    prompt_dict = {}
        
    if dataset_id == "asap":
        audio_path = posixpath.join(base_path, metadata_row['audio_performance'])
        midi_path = posixpath.join(base_path, metadata_row['midi_score'])
        
        alignment_path = audio_path.replace(".wav", "_corresp.txt")
        
        if not os.path.exists(alignment_path):
            assert("midi_score" in audio_path) # synthesis
            
            speed_midi_len = midi_len
            if "speed_ratio" in metadata_row: # alignment for the ones that's speed-augmented
                speed_midi_len = midi_len * metadata_row['speed_ratio']
                
            segpoints = np.arange(0, 1000, speed_midi_len)
        else:
            segpoints = asap_alignment_segpoint(alignment_path, midi_len=midi_len)
        
        
        if "midi_score" in audio_path:
            prompt_dict['perf'] = ", plain synthesis"
        else:
            prompt_dict['perf'] = ", expressive performance"
        
        prompt_dict['instrument'] = "Piano"
        
        if "midi_score" not in audio_path: 
            prompt_dict['title'] = ", ".join(metadata_row['audio_performance'].split("/")[:-1])
                
            
    if dataset_id == "yamahasax":
        if not metadata_row['align_success']:
            return None, None, None, None, None
    
        audio_path = posixpath.join(base_path, metadata_row['WAV Path'])
        align_midi_path = posixpath.join(base_path, metadata_row['MIDI Path'])
        midi_path = posixpath.join(base_path, metadata_row['Transposed MIDI Path'])
        
        alignment_path = posixpath.join(base_path, metadata_row['Alignment Path'])
        segpoints = get_yamahasax_segpoints(align_midi_path, alignment_path, midi_len=midi_len)
        
        path_parts = audio_path.split('/')
        form = path_parts[-4].split('_')[-1]  # Extracts 'BossaNova', 'Funk', etc.
        title = re.sub(r'\.mid|_Theme1|Sax_|Expressive_|No_expression_|On_', '', path_parts[-1]) 
        prompt_dict['perf'] = ('Expressive playing' if 'Expressive' in path_parts[-1] else 'No expression')
        
        prompt_dict['instrument'] = "Saxophone"
        prompt_dict['title'] = f"{form}, {title}"
    
    if dataset_id == "gaps":
        if "good_alignment" in metadata_row and (not metadata_row['good_alignment']):
            return None, None, None, None, None
        
        audio_path = posixpath.join(base_path, metadata_row['audio_path'])
        midi_path = posixpath.join(base_path, metadata_row['score_midi_path'])

        if  "alignment_file_dir" in metadata_row:
            if (isinstance(metadata_row['alignment_file_dir'], float)):
                return None, None, None, None, None
            alignment_dir_path = posixpath.join(base_path, metadata_row['alignment_file_dir'])
            
            segpoints = gaps_alignment_segpoint(alignment_dir_path, midi_len=midi_len)
        else:
            segpoints = np.arange(0, 1000, midi_len)
            
        prompt_dict['instrument'] = "Guitar"
        prompt_dict['perf'] = "Expressive performance"
    
    
    
    if dataset_id == "urmp":
        pass
        
    if dataset_id == "musicnet":
        audio_path = posixpath.join(base_path, metadata_row['audio_path'])
        midi_path = posixpath.join(base_path, metadata_row['score_midi_path'])
        
        if 'alignment_path'  in metadata_row:
            alignment_path = posixpath.join(base_path, metadata_row['alignment_path'])
            
            try:
                segpoints = get_musicnet_segpoints(midi_path, alignment_path, midi_len=midi_len)
            except:
                print(f"Error in {audio_path}")
                return None, None, None, None, None
        else:
            segpoints = np.arange(0, 1000, midi_len)
            
        prompt_dict['instrument'] = metadata_row['instrumentation'] if 'instrumentation' in metadata_row else ""
        prompt_dict['title'] = metadata_row['score_midi_path'].split("/")[1] + ", " + (metadata_row['piece'] if 'piece' in metadata_row else "")
        prompt_dict['perf'] = "Expressive performance"

    if dataset_id == "burgmuller":
        
        audio_path = posixpath.join(base_path, metadata_row['audio_path'])
        midi_path = posixpath.join(base_path, metadata_row['score_midi_path'])

        if  "alignment_file_dir" in metadata_row:
            if (isinstance(metadata_row['alignment_file_dir'], float)):
                return None, None, None, None, None
            alignment_dir_path = posixpath.join(base_path, metadata_row['alignment_file_dir'])
            
            segpoints = gaps_alignment_segpoint(alignment_dir_path, midi_len=midi_len) # same one, with parangonar
        else:
            segpoints = np.arange(0, 1000, midi_len)
            
        prompt_dict['instrument'] = "Piano"
        prompt_dict['perf'] = "Performance with mistakes"
        
    if dataset_id == 'vienna422':
        
        audio_path = posixpath.join(base_path, metadata_row['audio_path'])
        midi_path = posixpath.join(base_path, metadata_row['score_midi_path'])
        alignment_path = posixpath.join(base_path, metadata_row['alignment_path'])
        
        segpoints = get_vienna422_segpoints(alignment_path, midi_path, midi_len=midi_len)
        
        prompt_dict['instrument'] = "Piano"
        prompt_dict['perf'] = "Expressive performance"
        prompt_dict['title'] = metadata_row['audio_path'].split("/")[-2]
        prompt_dict['performer'] = metadata_row['performer_label']

    if dataset_id == 'bachviolin':
        
        audio_path = posixpath.join(base_path, metadata_row['audio_path'])
        midi_path = posixpath.join(base_path, metadata_row['score_midi_path'])
        alignment_path = posixpath.join(base_path, metadata_row['alignment_path'])
        notes_path = alignment_path.replace("alignments", "notes")
        
        segpoints = get_bachviolin_segpoints(alignment_path, notes_path, midi_path, midi_len=midi_len)
        
        prompt_dict['instrument'] = "Violin"
        prompt_dict['perf'] = "Expressive performance"
        prompt_dict['title'] = metadata_row['score_midi_path'].split("/")[-2]
        prompt_dict['performer'] = metadata_row['audio_path'].split("/")[-2]  
        
    if type(segpoints) == type(None):
        return None, None, None, None, None
    
    midi_tokens = midi_to_tokens(midi_path, codec, midi_length=midi_len)
    if len(midi_tokens) != len(segpoints) - 1:
        # print(f"Warning: Length mismatch at {audio_path}, tokens: {len(midi_tokens)}, segpoints: {len(segpoints)}")
        pass
    num_chunks = min(len(midi_tokens), len(segpoints) - 1)
    
    return audio_path, midi_tokens, segpoints, num_chunks, prompt_dict


SPEED_TIERS = {
    "very slow": ["twice as slow as", "significantly slower", "far slower", "about half the speed of"],
    "slow": ["considerably slower", "moving slower"],
    "slightly slow": ["a bit slower than score", "just under the score’s tempo", "slightly behind the intended pace"],
    "neutral": ["at the original speed", "faithfully matching", "in line with the score's tempo"],
    "slightly fast": ["a bit faster", "just above the score’s speed", "slightly faster than score"],
    "fast": ["notably faster", "well beyond the original tempo"],
    "very fast": ["almost twice as fast", "substantially quicker", "at a rapid pace, nearly double the tempo"]
}

def tier_speed_description(speed_ratio):
    """Returns a descriptive phrase based on speed ratio."""
    if speed_ratio >= 1.8:
        return random.choice(SPEED_TIERS["very slow"])
    elif 1.5 <= speed_ratio < 1.8:
        return random.choice(SPEED_TIERS["slow"])
    elif 1.2 <= speed_ratio < 1.5:
        return random.choice(SPEED_TIERS["slightly slow"])
    elif 0.8 <= speed_ratio <= 1.2 or (speed_ratio <= 0.1):
        return random.choice(SPEED_TIERS["neutral"])
    elif 0.6 <= speed_ratio < 0.8:
        return random.choice(SPEED_TIERS["slightly fast"])
    elif 0.4 <= speed_ratio < 0.6:
        return random.choice(SPEED_TIERS["fast"])
    else:
        return random.choice(SPEED_TIERS["very fast"])

def sample_prompt(prompt_dict, dummy_prompt=False):
    """
    prompt have several fields, perf and speed are required and the rest of the fields are sampled with 0.5 probability.
    Then they are concatenated with random order
    """
    
    if dummy_prompt:
        return "synthesis"

    required_fields = ['perf', 'speed']
    sampled_fields = {field: prompt_dict[field] for field in required_fields if field in prompt_dict}

    optional_fields = {
        field: prompt_dict[field]
        for field in prompt_dict
        if field not in required_fields and random.random() < 0.5
    }

    combined_fields = list(sampled_fields.values()) + list(optional_fields.values())

    random.shuffle(combined_fields)

    prompt = ", ".join(combined_fields)
    
    return prompt



class LocalDatasetConfig:
    def __init__(
        self,
        id: str,
        path: str,
        metadata_path: str,
        metadata_augmented_path: Optional[str] = None,
        custom_metadata_fn: Optional[Callable[[str], str]] = None
    ):
        self.id = id
        self.path = path
        self.metadata_path = metadata_path
        self.metadata_augmented_path = metadata_augmented_path
        self.custom_metadata_fn = custom_metadata_fn

class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        configs,
        sample_size=65536, 
        sample_rate=48000, 
        keywords=None, 
        random_crop=True,
        midi_length=10,
        force_channels="stereo"
    ):
        super().__init__()
        self.filenames = []

        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
        )

        self.root_paths = []

        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop)
        
        resolution = 100
        self.midi_length = midi_length
        self.codec = Codec(int(midi_length * resolution + 1))

        self.force_channels = force_channels

        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )

        self.sr = sample_rate

        self.custom_metadata_fns = {}

        for config in configs:
            self.root_paths.append(config.path)
            self.filenames.extend(get_audio_filenames(config.path, keywords))
            if config.custom_metadata_fn is not None:
                self.custom_metadata_fns[config.path] = config.custom_metadata_fn

        print(f'Found {len(self.filenames)} files')
        

    def load_file(self, filename):
        ext = filename.split(".")[-1]

        if ext == "mp3":
            with AudioFile(filename) as f:
                audio = f.read(f.frames)
                audio = torch.from_numpy(audio)
                in_sr = f.samplerate
        else:
            audio, in_sr = torchaudio.load(filename, format=ext)

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr)
            audio = resample_tf(audio)

        return audio

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
            audio_filename = self.filenames[idx]
        # try:

            info = {}

            info["path"] = audio_filename

            for root_path in self.root_paths:
                if root_path in audio_filename:
                    info["relpath"] = path.relpath(audio_filename, root_path)

            for custom_md_path in self.custom_metadata_fns.keys():
                if custom_md_path in audio_filename:
                    custom_metadata_fn = self.custom_metadata_fns[custom_md_path]
                    custom_metadata = custom_metadata_fn(info)
                    info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    return self[random.randrange(len(self))]                

            start_time = time.time()
            audio = self.load_file(audio_filename)

            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio )

            info["timestamps"] = (t_start, t_end)
            info["seconds_start"] = seconds_start
            info["seconds_total"] = seconds_total
            info["padding_mask"] = padding_mask

            # save_midi_and_wav(idx, 0, info["midi"], self.codec, audio, self.sr)

            # Run augmentations on this sample (including random crop)
            if self.augs is not None:
                audio = self.augs(audio)

            audio = audio.clamp(-1, 1)

            # Encode the file to assist in prediction
            if self.encoding is not None:
                audio = self.encoding(audio)
                
            end_time = time.time()
            info["load_time"] = end_time - start_time            

            return (audio, info)
        # except Exception as e:
        #     print(f'Couldn\'t load file {audio_filename}: {e}')
        #     return self[random.randrange(len(self))]




class SampleDatasetChunk(torch.utils.data.Dataset):
    def __init__(
        self, 
        configs,
        sample_size=65536, 
        sample_rate=48000, 
        keywords=None, 
        random_crop=True,
        midi_length=10,
        force_channels="stereo",
        split="train",
        dummy_prompt=False
    ):
        super().__init__()
        
        filenames = []

        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
        )

        self.root_paths = []

        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop)
        
        resolution = 100
        self.midi_length = midi_length
        self.codec = Codec(int(midi_length * resolution + 1))

        self.force_channels = force_channels

        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )

        self.sr = sample_rate
        self.dummy_prompt = dummy_prompt

        # load from the segmentation points to determine the boundaries of segments
        self.boundaries = [0]
        self.data_list = []

        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        for config in configs:
            self.root_paths.append(config.path)

            metadata = pd.read_csv(config.metadata_path)   
            
            if split == 'train':
                metadata = metadata[metadata['split'] == 'train']
            else:
                if 'split' in metadata.columns:
                    metadata = metadata[metadata['split'] != 'train']
            
            meta_iter = metadata.iterrows()
            
            if config.metadata_augmented_path is not None:
                metadata_augmented = pd.read_csv(config.metadata_augmented_path)
                if split == 'train':
                    metadata_augmented = metadata_augmented[metadata_augmented['split'] == 'train']
                else:
                    metadata_augmented = metadata_augmented[metadata_augmented['split'] != 'train']
                
                meta_iter = chain(metadata.iterrows(), metadata_augmented.iterrows())
                                            
            for id, row in meta_iter:
                try:
                    audio_path, midi_tokens, segpoints, num_chunks, prompt_dict = get_metadata_info(config.id, config.path, row, self.codec, midi_len=self.midi_length)
                except Exception as e:
                    print(f'Error in {config.id}, {config.path}, {row}: {e}')
                    continue

                
                if type(segpoints) == type(None):
                    continue
                
                self.boundaries.append(self.boundaries[-1] + num_chunks)
                self.data_list.append(
                    (audio_path, midi_tokens, segpoints, prompt_dict)
                ) 
                # print(len(self.data_list)-1, audio_path)
                
        
        print(f'Found {len(self.data_list)} files, total number of chunks: {self.boundaries[-1]}')    

    def load_file(self, filename):
        ext = filename.split(".")[-1]

        if ext == "mp3":
            with AudioFile(filename) as f:
                audio = f.read(f.frames)
                audio = torch.from_numpy(audio)
                in_sr = f.samplerate
        else:
            audio, in_sr = torchaudio.load(filename, format=ext)

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr)
            audio = resample_tf(audio)

        return audio

    def __len__(self):
        return self.boundaries[-1]

    def _get_file_idx_and_chunk_idx(self, index: int):
        bin_pos = np.digitize(index, self.boundaries[1:], right=False)
        chunk_index = index - self.boundaries[bin_pos]
        return bin_pos, chunk_index
    

    def __getitem__(self, idx):
            file_idx, chunk_idx = self._get_file_idx_and_chunk_idx(idx)
            # chunk_idx = 0
            audio_path, midi_tokens, segpoints, prompt_dict = self.data_list[file_idx]
        # try:

            info = {}

            info["path"] = audio_path
            info['midi'] = midi_tokens[chunk_idx]
            optional_start_seconds, optional_end_seconds = segpoints[chunk_idx], segpoints[chunk_idx + 1]

            # if ('bwv_868' in info["path"]) :
            #     print(f'bwv_868: {optional_start_seconds}, {optional_end_seconds}')
            #     hook()

            if "__reject__" in info and info["__reject__"]:
                return self[random.randrange(len(self))]                

            start_time = time.time()

            try:
                audio = self.load_file(audio_path)
                audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio, optional_start_seconds, optional_end_seconds )
            except:
                return self[random.randrange(len(self))]

            info["timestamps"] = (t_start, t_end)
            info["seconds_start"] = seconds_start
            info["seconds_total"] = seconds_total
            info["padding_mask"] = padding_mask
            
            # update the text prompt based on speed ratio
            prompt_dict['speed'] = tier_speed_description(seconds_total / self.midi_length)
            info['prompt'] = sample_prompt(prompt_dict, dummy_prompt=self.dummy_prompt)

            # if random.random() < 0.01:
            #     print(info['prompt'])
            #     save_midi_and_wav(file_idx, chunk_idx, info["midi"], self.codec, audio, self.sr)
            save_midi_and_wav(audio_path.split("/")[-1][:-4], chunk_idx, info["midi"], self.codec, audio, self.sr)
            
                
            info['file_idx'] = file_idx
            info['chunk_idx'] = chunk_idx
            info['audio_path'] = audio_path

            # Run augmentations on this sample (including random crop)
            if self.augs is not None:
                audio = self.augs(audio)

            audio = audio.clamp(-1, 1)

            # Encode the file to assist in prediction
            if self.encoding is not None:
                audio = self.encoding(audio)
                
            end_time = time.time()
            info["load_time"] = end_time - start_time            

            return (audio, info)
        # except Exception as e:
        #     print(f'Couldn\'t load file {audio_filename}: {e}')
        #     return self[random.randrange(len(self))]


def midi_to_tokens(midi_path, codec, midi_length=10):
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    ns = note_seq.midi_file_to_note_sequence(midi_path)
    ns = note_seq.apply_sustain_control_changes(ns)
    
    # get the first 10 seconds of midi 
    tokens, _ = preprocess( ns, segment_length=midi_length, codec=codec)

    return tokens




def audio_decoder(key, value):
    # Get file extension from key
    ext = key.split(".")[-1]

    if ext in AUDIO_KEYS:
        return torchaudio.load(io.BytesIO(value))
    else:
        return None

def collation_fn(samples):
        batched = list(zip(*samples))
        result = []
        for b in batched:
            if isinstance(b[0], (int, float)):
                b = np.array(b)
            elif isinstance(b[0], torch.Tensor):
                b = torch.stack(b)
            elif isinstance(b[0], np.ndarray):
                b = np.array(b)
            else:
                b = b
            result.append(b)
        return result


def create_dataloader_from_config(dataset_config, batch_size, sample_size, sample_rate, audio_channels=2, num_workers=4, split="train"):

    dataset_type = dataset_config.get("dataset_type", None)

    assert dataset_type is not None, "Dataset type must be specified in dataset config"

    if audio_channels == 1:
        force_channels = "mono"
    else:
        force_channels = "stereo"

    if dataset_type == "audio_dir":

        audio_dir_configs = dataset_config.get("datasets", None)

        assert audio_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

        configs = []

        for audio_dir_config in audio_dir_configs:
            audio_dir_path = audio_dir_config.get("path", None)
            assert audio_dir_path is not None, "Path must be set for local audio directory configuration"

            custom_metadata_fn = None
            custom_metadata_module_path = audio_dir_config.get("custom_metadata_module", None)

            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)                

                custom_metadata_fn = metadata_module.get_custom_metadata

            configs.append(
                LocalDatasetConfig(
                    id=audio_dir_config["id"],
                    path=audio_dir_path,
                    metadata_path=audio_dir_config.get("metadata_path", None),
                    metadata_augmented_path=audio_dir_config.get("metadata_augmented_path", None),
                    custom_metadata_fn=custom_metadata_fn
                )
            )

        use_chunks = dataset_config.get("use_chunks", False)
        if use_chunks:
            train_set = SampleDatasetChunk(
                configs,
                sample_rate=sample_rate,
                sample_size=sample_size,
                random_crop=dataset_config.get("random_crop", True),
                midi_length=dataset_config.get("midi_length", 10),
                force_channels=force_channels,
                split=split,
                dummy_prompt=dataset_config.get("dummy_prompt", False)
            )
        else:
            train_set = SampleDataset(
                configs,
                sample_rate=sample_rate,
                sample_size=sample_size,
                random_crop=dataset_config.get("random_crop", True),
                midi_length=dataset_config.get("midi_length", 10),
                force_channels=force_channels,
                split=split
            )

        return torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                num_workers=num_workers, persistent_workers=True, pin_memory=True, drop_last=True, collate_fn=collation_fn)

    elif dataset_type in ["s3", "wds"]: # Support "s3" type for backwards compatibility
        wds_configs = []

        for wds_config in dataset_config["datasets"]:

            custom_metadata_fn = None
            custom_metadata_module_path = wds_config.get("custom_metadata_module", None)

            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)                

                custom_metadata_fn = metadata_module.get_custom_metadata

            if "s3_path" in wds_config:

                wds_configs.append(
                    S3DatasetConfig(
                        id=wds_config["id"],
                        s3_path=wds_config["s3_path"],
                        custom_metadata_fn=custom_metadata_fn,
                        profile=wds_config.get("profile", None),
                    )
                )
            
            elif "path" in wds_config:
                    
                    wds_configs.append(
                        LocalWebDatasetConfig(
                            id=wds_config["id"],
                            path=wds_config["path"],
                            custom_metadata_fn=custom_metadata_fn
                        )
                    )

        return WebDatasetDataLoader(
            wds_configs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            batch_size=batch_size,
            random_crop=dataset_config.get("random_crop", True),
            num_workers=num_workers,
            persistent_workers=True,
            force_channels=force_channels,
            epoch_steps=dataset_config.get("epoch_steps", 2000)
        ).data_loader