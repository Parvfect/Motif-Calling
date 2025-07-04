from nn import CallerEmpirical
import torch
from decoder import GreedyCTCDecoder
import os
from tqdm import tqdm
from sklearn.preprocessing import normalize
from torch.nn.utils.rnn import pad_sequence
from utils import sort_transcript, detect_reverse_oriented_read
import pandas as pd
from typing import List
from fast5_input import extract_fast5_data_from_file
from typing import List, Tuple


n_classes = 19
hidden_size = 256

def load_model(model_path, device):
    if device == torch.device('cpu'):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(model_path)

    model = CallerEmpirical(num_classes=n_classes, hidden_dim=hidden_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def model_init(fast5_path: str):

    squiggles, read_ids = extract_fast5_data_from_file(fast5_filepath=fast5_path)
    forward_model_path = 'forward.pth'
    mixed_model_path = 'mixed.pth'
    reverse_model_path = 'reverse.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print(f"Running on {device}")

    # Initialising decoder
    greedy_decoder = GreedyCTCDecoder(n_classes=19)

    forward_model = load_model(forward_model_path, device=device)
    mixed_model = load_model(mixed_model_path, device=device)
    reverse_model = load_model(reverse_model_path, device=device)

    return squiggles, read_ids, forward_model, mixed_model, reverse_model, device, greedy_decoder

def model_inference(
        data_arr: List[List[int]], read_ids: List[str],
        forward_model: torch.nn, mixed_model: torch.nn,
        reverse_model: torch.nn, device: torch.device,
        greedy_decoder: GreedyCTCDecoder) -> Tuple[
            List[List[int]], List[str], List[str]]:

    greedy_transcripts_arr = []
    sorted_greedy_transcripts = []
    read_ids_arr = []
    qualities = []
    full_qs = []

    n_training_samples = len(data_arr)

    print(f"Inference on {n_training_samples} squiggles")

    batch_size = 8
    forward_model.to(device)
    reverse_model.to(device)
    mixed_model.to(device)

    with torch.no_grad():
        for ind in tqdm(range(0, n_training_samples, batch_size)):

            if n_training_samples - ind < batch_size:
                batch_size = n_training_samples - ind - 1
                #continue
            
            input_seqs = [
                normalize([data_arr[k]], norm='max').flatten() for k in range(ind, ind + batch_size)]
            

            input_seqs = pad_sequence([torch.tensor(
                        i, dtype=torch.float32) for i in input_seqs], batch_first=True)
            
            input_seqs = input_seqs.view(input_seqs.shape[0], 1, input_seqs.shape[1])
            input_seqs = input_seqs.to(device)

            try:
                forward_model_output = forward_model(input_seqs)
                reverse_model_output = reverse_model(input_seqs)
                mixed_model_output = mixed_model(input_seqs)
                
                for k in range(batch_size):

                    greedy_result_mixed, quality, full_q = greedy_decoder(
                        mixed_model_output[k])

                    if detect_reverse_oriented_read(greedy_result_mixed):
                        greedy_result, quality, full_q = greedy_decoder(
                        reverse_model_output[k])
                        greedy_transcript = " ".join(greedy_result)
                    else:
                        greedy_result, quality, full_q = greedy_decoder(
                        forward_model_output[k])
                        greedy_transcript = " ".join(greedy_result)

                    sorted_greedy = sort_transcript(greedy_transcript)

                    greedy_transcripts_arr.append(greedy_transcript)
                    sorted_greedy_transcripts.append(sorted_greedy)
                    qualities.append(quality)
                    full_qs.append(full_q)

                torch.cuda.empty_cache()
                if device == torch.device('cuda'):
                    del input_seqs
                
                read_ids_arr.extend(read_ids[ind: ind + batch_size])

            except Exception as e:
                print(f"Ignoring error {e} and continuing inference")
                continue

    return sorted_greedy_transcripts, greedy_transcripts_arr, read_ids_arr, qualities, full_qs