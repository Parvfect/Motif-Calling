
from typing import List


def sort_transcript(transcript: str, payload: bool = False) -> List[List[int]]:
    """Convert model output transcript into a cycle level prediction"""

    if payload:
        cycles = [[] for i in range(10)]
        cutoff_motif = 8
        starting_pos = 9
    else:
        cycles = [[] for i in range(8)]
        cutoff_motif = 10
        starting_pos = 11
    
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
                if split_transcript[i-1] > cutoff_motif:
                    cycle_number = split_transcript[i-1] - starting_pos
                    cycles[cycle_number].append(split_transcript[i])

                # Checking for Forward Spacer
                elif i < len(split_transcript) - 1:
                    if split_transcript[i+1] > cutoff_motif:
                        cycle_number = split_transcript[i+1] - starting_pos
                        cycles[cycle_number].append(split_transcript[i])

            else:
                if i < len(split_transcript) - 1:
                    # Checking for Forward Spacer
                    if split_transcript[i+1] > cutoff_motif:
                        cycle_number = split_transcript[i+1] - starting_pos
                        cycles[cycle_number].append(split_transcript[i])   
    
    return [list(set(i)) for i in cycles]


def get_motifs_identified(sorted_payload_transcript: List[List[int]]):
    """Converts the sorted transcript into HW format of ltm8_ixj for motifs found in a read"""

    motifs_found = []

    cycle_number = 1
    for i in sorted_payload_transcript:
        for j in i:
            motifs_found.append(f'ltm8_{cycle_number}x{j}')
        cycle_number += 1

    return motifs_found

def detect_reverse_oriented_read(transcript_reverse_model):

    # Remove payloads, make set, and check that each spacer motif is increasing
    no_payload_transcript = [
        int(i) for i in transcript_reverse_model if int(i) > 8]
    
    if len(no_payload_transcript) == 0:
        return False

    first_spacer = no_payload_transcript[0]
    flag = False
    for j in no_payload_transcript[1:]:
        if not j == first_spacer:
            if j < first_spacer:
                flag = True
            else:
                flag = False
            break
        
    return flag

    