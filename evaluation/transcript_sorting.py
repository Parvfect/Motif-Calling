
from typing import List
import math


def sort_transcript(transcript):

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


def sort_transcript_with_address(transcript):

    cycles = [[] for i in range(10)]
    
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
                if split_transcript[i-1] > 8:
                    cycle_number = split_transcript[i-1] - 9
                    cycles[cycle_number].append(split_transcript[i])

                # Checking for Forward Spacer
                elif i < len(split_transcript) - 1:
                    if split_transcript[i+1] > 8:
                        cycle_number = split_transcript[i+1] - 9
                        cycles[cycle_number].append(split_transcript[i])

            else:
                if i < len(split_transcript) - 1:
                    # Checking for Forward Spacer
                    if split_transcript[i+1] > 8:
                        cycle_number = split_transcript[i+1] - 9
                        cycles[cycle_number].append(split_transcript[i])   
    return [list(set(i)) for i in cycles]

def create_reduced_spacer_transcript(motif_seq: List[int]) -> List[int]:
    """ 12 4 12 12 3 12 -> 12 4 2 3 4 12 13 2 4 5 3 13"""

    seq = []
    cycle_transcript = sort_transcript(
        " ".join([str(i) for i in motif_seq]))

    for ind, i in enumerate(cycle_transcript):
        if len(i) == 0:
            continue
        
        seq.append(ind + 11)
        seq.extend(list(set(i)))
        seq.append(ind + 11)

    return seq
    



def sort_transcript_reduced_spacers(transcript):
    " 12 4 3 5 4 12 4 5 6 13 "

    if not type(transcript) == str:
        transcript = " ".join([str(i) for i in transcript])

    split_transcript = transcript.split()
    sorted_transcript = [[] for i in list(range(8))]
    flag = False
    cycle_number = 0

    for i in split_transcript[::-1]:
        if not i == ' ':
            if int(i) > 8:
                if not i == cycle_number:
                    sorted_transcript
                    cycle_number = int(i) - 9
                else:
                    continue
            else:
                if cycle_number >= len(sorted_transcript):
                    cycle_number -= 2
                sorted_transcript[cycle_number].append(int(i))

    return sorted_transcript
        