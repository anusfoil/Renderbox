import os
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