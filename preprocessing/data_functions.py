
from ont_fast5_api.fast5_interface import get_fast5_file
import pandas as pd


def get_data_from_fast5(fast5_filepath: str, read_ids_db=None):
    raw_data_arr = []
    read_ids = []
    with get_fast5_file(fast5_filepath, mode="r") as f5:
        for read in f5.get_reads():
            if read_ids_db:  # Extracting only those reads that are within the read_ids
                if not read.read_id in read_ids_db:
                    continue
            raw_data = read.get_raw_data()
            raw_data_arr.append(raw_data)
            read_ids.append(read.read_id)
    return raw_data_arr, read_ids


def get_cleaned_encoded_file(encoded_df, address=False):

    # Joining payloads
    payload_cols = [col for col in encoded_df.columns if col.startswith('Payload')]
    encoded_df['payload'] = encoded_df[payload_cols].astype(str).agg(', '.join, axis=1)
    encoded_df['payload'] = encoded_df['payload'].apply(lambda x: list(eval(x)))

    if address:
        # Fixing addresses
        encoded_df['Address_Incrementer_1'] = encoded_df['Address_Incrementer_1'].apply(lambda x: f'barcode_external0{x[1]}')
        encoded_df['Address_Incrementer_2'] = encoded_df['Address_Incrementer_2'].apply(lambda x: f'_internal0{x[1]}')
        address_cols = [col for col in encoded_df.columns if col.startswith('Address')]
        encoded_df['HW_Address'] = encoded_df[address_cols].astype(str).agg(''.join, axis=1)

    # Selecting important columns
    encoded_df = encoded_df[['HW_Address', 'payload', 'ONT_Barcode']]
    return encoded_df


def library_motif_to_sequence(library_motif_preds):
    """Converts motif search predictions to sorted sequences"""
    return


def create_spacer_sequence(payload_prediction):
    """Returns spacer sequence from the payload cycle level prediction (for training)"""
    return

def sort_transcript(transcript):
    """Create payload level prediction from the transcript """

    cycles = [[] for i in range(8)]
    
    if type(transcript) == str:
        transcript = transcript.split()
    
    split_transcript = [int(i) for i in transcript if i != '']
    
    for i in range(len(split_transcript)):

        found_motif = split_transcript[i]

        # If we have a payload motif
        if found_motif < 9:

            # finding the spacers - only for payload cycles
            if i > 0:
                # Checking for Back Spacer
                if split_transcript[i-1] > 10:
                    cycle_number = split_transcript[i-1] - 11
                    cycles[cycle_number].append(split_transcript[i])

                # Checking for Forward Spacer
                elif i < len(split_transcript) - 1:
                    if split_transcript[i+1] > 10:
                        cycle_number = split_transcript[i+1] - 11
                        cycles[cycle_number].append(split_transcript[i])

            else:
                if i < len(split_transcript) - 1:
                    # Checking for Forward Spacer
                    if split_transcript[i+1] > 10:
                        cycle_number = split_transcript[i+1] - 11
                        cycles[cycle_number].append(split_transcript[i])   
    return [list(set(i)) for i in cycles]



def sort_library_motif_transcript(library_prediction):
    """Sorts library motif transcripts while fixing for typos, by looking at the payloads to get the best orientation matches"""

    sorted_prediction = [[] for i in range(10)]

    split_library_prediction = library_prediction.split('|')

    #print(split_library_prediction)
    for i in split_library_prediction:

        if len(i) < 8:
            continue
        # searching for the 10
        motif_found = int(i[-1])
        cycle_address = None  # starts from 1 there and from 0 here
        if i[-4] == '1':
            if i[-3] == '0':
                cycle_address = 10
                sorted_prediction[cycle_address - 1].append(motif_found)
        elif i[-4] == '9':
            continue
        else:
            cycle_address = int(i[-3])
            sorted_prediction[cycle_address - 1].append(motif_found)

    return sorted_prediction
    