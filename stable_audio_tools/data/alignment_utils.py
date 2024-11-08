import os, re
import mido
import numpy as np
import pandas as pd
import partitura as pt
from scipy.interpolate import interp1d


def load_alignment(match_file):

    pna, sna, alignment  = pt.load_nakamuracorresp(match_file)   
    return pna, sna, alignment 


def asap_alignment_segpoint(match_file, midi_len=5.12):
    """ 
    (removed hdf5 since each time the midi_len could be different)
    Args:
        match_file (str): path to the alignment file
            for asap, it's the parangonar note_alignment.tsv file
    """        

    if not os.path.exists(match_file) and "midi_score" in match_file:
        return np.arange(0, 1000, midi_len)

    pna, sna, alignment = load_alignment(match_file)
    
    score_df = pd.DataFrame(sna)
    performance_df = pd.DataFrame(pna)
    alignment_df = pd.DataFrame(alignment)

    if len(alignment_df[alignment_df['label'] != 'match']) > len(alignment_df[alignment_df['label'] == 'match']):
        print(f"Warning: {match_file} has more non-matching labels than matching labels") 
        return None

    alignment_df = alignment_df[alignment_df['label'] == 'match']

    merged_df = pd.merge(alignment_df, score_df, left_on='score_id', right_on='id', suffixes=('_score', '_perf'))
    merged_df = pd.merge(merged_df, performance_df, left_on='performance_id', right_on='id', suffixes=('_score', '_perf'))

    columns_to_keep = [col for col in merged_df.columns if (not col.startswith('id_') and not col.endswith('_div') and not col.endswith('_quarter'))]
    merged_df = merged_df[columns_to_keep]
        
    interp_func = interp1d(merged_df['onset_sec_score'], merged_df['onset_sec_perf'], fill_value='extrapolate', kind='nearest')
    
    # segpoints is the interpolated performance time of each midi_len seconds of the score time
    segpoints = interp_func(np.arange(0, interp_func.x[-1] + midi_len, midi_len))

    if np.isnan(segpoints[0]):
        segpoints[0] = 0  # the first one might be zero becuase the perf starts later

    assert np.isnan(segpoints).sum() == 0, "segpoints doesn't contains NaNs"
    assert np.isinf(segpoints).sum() == 0, "segpoints doesn't contains Infs"

    return segpoints

########################### YAMAHASAX ###########################

def parse_txt_file(txt_file):
    """Parses the .txt file to extract note onsets and offsets in terms of frames."""
    audio_onsets = []
    audio_offsets = []
    pitch_classes = []  
    note_lengths = []  # Symbolic lengths

    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # Filter out lines with separators
    lines = [line for line in lines if '-----------------------' not in line]

    for i, line in enumerate(lines[:-1]): # assumption: last line is a different format and not needed
        if "Frame" in line and "~:" in line and "length" in line:
            parts = line.strip().split('~:')
            frame_number = int(parts[0].split()[1])
            note_info = parts[1].strip().split()

            # TODO: there are cases with slurs and other symbols, need to handle them
            if len(note_info) >= 3 and re.match("[+-]?\d", note_info[0]):
                
                pitch_class = note_info[0]  
                length = float(note_info[-1])  # Symbolic note length (from the score)

                # Onset frame for the current note
                onset_frame = frame_number
                # Determine offset frame: next onset or 'Unvoiced'
                # Search for the next valid frame with note information
                next_line = lines[i + 1]
                next_frame_number = int(next_line.split('~:')[0].split()[1])
                offset_frame = next_frame_number

                pitch_classes.append(pitch_class)
                audio_onsets.append(onset_frame)
                audio_offsets.append(offset_frame)
                note_lengths.append(length)
                

    return pitch_classes, audio_onsets, audio_offsets, note_lengths


def parse_midi_file(midi_file):
    """Parses the MIDI file to extract note onsets and offsets in terms of MIDI time."""
    midi_onsets = []
    midi_offsets = []
    pitches = []
    note_lengths = []

    mid = mido.MidiFile(midi_file)
    current_time = 0
    notes_on = {}

    for msg in mid:
        current_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            notes_on[msg.note] = current_time
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            # Note offset
            if msg.note in notes_on:
                pitches.append(msg.note)
                midi_onsets.append(notes_on[msg.note])
                midi_offsets.append(current_time)
                note_lengths.append(current_time - notes_on[msg.note])  # MIDI note duration
        
    return pitches, midi_onsets, midi_offsets, note_lengths


def align_notes(txt_file, midi_file, frame_to_samples=1200, sample_rate=96000, verbal=False):
    """Aligns MIDI and audio onsets/offsets based on symbolic lengths and time."""
    audio_pitch_classes, audio_onsets, audio_offsets, symbolic_lengths = parse_txt_file(txt_file)

    midi_pitches, midi_onsets, midi_offsets, midi_lengths = parse_midi_file(midi_file)

    if len(midi_onsets) != len(audio_onsets):
        print(txt_file)
        print("Warning: Mismatch in the number of notes between audio and MIDI")
        raise ValueError("Mismatch in the number of notes between audio and MIDI")

    audio_onsets_seconds = [frame * frame_to_samples / sample_rate for frame in audio_onsets]
    audio_offsets_seconds = [frame * frame_to_samples / sample_rate for frame in audio_offsets]

    aligned_midi_times = []
    aligned_audio_times = []

    # expectation ratio
    for i in range(len(midi_onsets)):
        if symbolic_lengths[i] == 0:
            continue
        ratio = symbolic_lengths[i] / midi_lengths[i]
        break

    for i in range(len(midi_onsets)):
        aligned_midi_times.extend([midi_onsets[i], midi_offsets[i]])  
        aligned_audio_times.extend([audio_onsets_seconds[i], audio_offsets_seconds[i]])  

        # Check symbolic length consistency (assumes stable tempo between MIDI and audio)
        if i < len(midi_onsets) - 1:
            symbolic_length = symbolic_lengths[i]
            midi_length = midi_lengths[i]

            # Verify if the MIDI length matches the symbolic length (within a tolerance)
            expected_midi_duration = symbolic_length / ratio
            if abs(midi_length - expected_midi_duration) > 0.05:  # Allow a small tolerance
                if verbal:
                    print(f"Warning: Length mismatch at note {i}, symbolic {symbolic_length} vs. MIDI length {expected_midi_duration}")
            
            midi_pitch_class = midi_pitches[i] % 12
            audio_pitch_class =  (int(audio_pitch_classes[i]) - 3) % 12
            if midi_pitch_class != audio_pitch_class:  # transposing to C
                if verbal:
                    print(f"Warning: Pitch mismatch at note {i}, MIDI {midi_pitch_class} vs. audio {audio_pitch_class}")

    alignment_func = interp1d(aligned_midi_times, aligned_audio_times, fill_value='extrapolate', kind='nearest')
    
    # if str(midi_file) == '/mnt/data1/intern-2024-04/yamahasax/20200724_AltoSax_Suzuki-Kei-san_BossaNova/midi/Expressive_Theme1/C-dur_in_Bossa.mid':
    #     hook()
    
    return alignment_func

def get_yamahasax_segpoints(midi_file, alignment_file, frame_to_samples=1200, sample_rate=96000, midi_len=5.12, verbal=False):
    
    alignment_func = align_notes(alignment_file, midi_file, frame_to_samples, sample_rate, verbal=verbal)
    segpoints = alignment_func(np.arange(0, alignment_func.x[-1] + midi_len, midi_len))
    
    assert np.isnan(segpoints).sum() == 0, "segpoints doesn't contains NaNs"
    assert np.isinf(segpoints).sum() == 0, "segpoints doesn't contains Infs"
    
    return segpoints