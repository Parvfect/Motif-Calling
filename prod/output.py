
import pandas as pd
from utils import sort_transcript, get_motifs_identified
from datetime import datetime
import os
import uuid


def save_inference_to_csv(
        sorted_greedy_transcripts, greedy_transcripts_arr, read_ids_arr,
        qualities, full_qs, savepath, fast5_filepath):

    motifs_identified = [
        get_motifs_identified(sort_transcript(i, payload=True)) for i in greedy_transcripts_arr]
    
    df = pd.DataFrame(
        {'read_id': read_ids_arr, 'payload_prediction': sorted_greedy_transcripts,
        'library_motif': motifs_identified, 'raw_transcript': greedy_transcripts_arr, "quality": qualities, "full_q": full_qs})

    fast5_file = os.path.basename(fast5_filepath)[:-6]
    uid = str(uuid.uuid4())
    full_save_path = os.path.join(savepath, f"caller_inference_{fast5_file}_{uid}.csv")

    df.to_csv(full_save_path)
    print(f"Output file saved to {full_save_path}")