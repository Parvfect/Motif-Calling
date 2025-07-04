
from nn import MotifCaller, NaiveCaller
from training_data import (
    load_training_data, data_preproc,
    create_label_for_training
)
from Levenshtein import ratio
from sklearn.preprocessing import normalize
import pandas as pd
from greedy_decoder import GreedyCTCDecoder
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import CTCLoss
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils import get_actual_transcript, get_savepaths, load_model
from transcript_sorting import sort_transcript, sort_transcript_reduced_spacers
from evaluation import evaluate_cycle_prediction
import numpy as np
from typing import List, Dict
import random
from model_config import ModelConfig
from training_monitoring import wandb_login, start_wandb_run
import wandb
from torch.nn.utils.rnn import pad_sequence


def run_epoch(
        model: MotifCaller, model_config: ModelConfig, optimizer: torch.optim, decoder: any, 
        X: List[torch.tensor], y: List[List[int]], ctc: CTCLoss, train: bool = False,
        display: bool = False, windows: bool = True, normalize_flag: bool = False, batch_size:int = 1) -> Dict[str, any]:

    n_training_samples = len(X)
    losses = np.zeros(n_training_samples)
    edit_distance_ratios = []
    greedy_transcripts = []
    motifs_found_arr = []
    motif_errs_arr = []
    display_iterations = int(n_training_samples / 2)
    device = model_config.device

    if train:
        model.train()
    else:
        model.eval()

    # Generate a random permutation of indices
    indices = np.random.permutation(len(X))

    # Apply the permutation to both X and y
    X = [X[ind] for ind in indices]
    y = [y[ind] for ind in indices]
    
    for ind in tqdm(range(n_training_samples)):
        
        if train:
            optimizer.zero_grad()

        # deal with batch stuff here
        input_sequence = X[ind]
        target_sequence = torch.tensor(y[ind]).to(device)

        if windows:
            input_sequence = input_sequence.to(device)

        else:
            if normalize_flag:
                input_sequence = normalize(
                    [input_sequence], norm='max')
            
            input_sequence = torch.tensor(
                input_sequence, dtype=torch.float32)
            input_sequence = input_sequence.view(1, 1, len(X[ind])).to(device)

        model_output = model(input_sequence)
        model_output = model_output.permute(1, 0, 2)  # Assuming log probs are computed in network
        
        if windows:
           #print(model_output.shape)
           model_output = model_output.reshape(
               model_output.shape[0] * model_output.shape[1],
                1, model_config.n_classes)
           #print(model_output.shape)

        model_output_flattened = model_output.view(
            model_output.shape[0] * model_output.shape[1], model_config.n_classes)
        n_timesteps = model_output.shape[0]
        input_lengths = torch.tensor([n_timesteps])
        label_lengths = torch.tensor([len(target_sequence)])
        
        if ind == 1:
            print(f"\n{n_timesteps/len(target_sequence)}")

        loss = ctc(
            log_probs=model_output, targets=target_sequence, input_lengths=input_lengths, target_lengths=label_lengths)
        
        
        if train:
            try:
                loss.backward()
            except Exception as e:
                print(f"Exception {e}")
                print(target_sequence)
                continue
            optimizer.step()
        
        losses[ind] = loss.item()
        
        greedy_result = decoder(model_output)
        greedy_transcript = " ".join(greedy_result)
        actual_transcript = get_actual_transcript(y[ind])
        sorted_greedy = sort_transcript(greedy_transcript)
        sorted_actual = sort_transcript(actual_transcript)
        motifs_found, motif_errs = evaluate_cycle_prediction(
            sorted_greedy, sorted_actual)
        edit_distance_ratios.append(ratio(greedy_transcript, actual_transcript))
        motifs_found_arr.append(motifs_found)
        motif_errs_arr.append(motif_errs)

        #print(f"\n{ratio_labels} aah {len(y[ind])}")
        
        if ind % 100 == 0 and not ind == 0:
            print(f"\nGreedy Transcript \n{greedy_transcript}")
            print(f"Actual Transcript \n{actual_transcript}")
            print(f"Cycle Greedy \n{sorted_greedy}")
            print(f"Cycle Actual \n{sorted_actual}")
            print(input_sequence[0][0][:15])

        
    return {
        "model": model,
        "losses": losses,
        "edit_distance_ratios": edit_distance_ratios,
        "motifs_found": motifs_found_arr,
        "motif_errs": motif_errs_arr
    }



def run_epoch_batched(
        model: MotifCaller, model_config: ModelConfig, optimizer: torch.optim, decoder: any, 
        X: List[torch.tensor], y: List[List[int]], ctc: CTCLoss, train: bool = False,
        display: bool = False, windows: bool = True, normalize_flag: bool = False, batch_size:int = 1) -> Dict[str, any]:

    n_training_samples = len(X)
    losses = np.zeros(int(n_training_samples / batch_size))
    edit_distance_ratios = []
    greedy_transcripts = []
    motifs_found_arr = []
    motif_errs_arr = []
    display_iterations = int(n_training_samples / 2)
    device = model_config.device


    if train:
        model.train()
    else:
        model.eval()
    
    # Generate a random permutation of indices
    indices = np.random.permutation(len(X))

    # Apply the permutation to both X and y
    X = [X[ind] for ind in indices]
    y = [y[ind] for ind in indices]

    for ind in tqdm(range(0, n_training_samples, batch_size)):

        if n_training_samples - ind < batch_size:
            continue
        
        if train:
            optimizer.zero_grad()

        if normalize_flag:
            input_seqs = [
                normalize([X[k]], norm='max').flatten() for k in range(ind, ind + batch_size)]
        else:
            input_seqs = X[ind: ind + batch_size]

        target_seqs = y[ind: ind + batch_size]

        input_seqs = pad_sequence([torch.tensor(
                    i, dtype=torch.float32) for i in input_seqs], batch_first=True)
        target_seqs = pad_sequence([torch.tensor(
                    i, dtype=torch.float32) for i in target_seqs], batch_first=True).to(device)
        
        input_seqs = input_seqs.view(input_seqs.shape[0], 1, input_seqs.shape[1]).to(device)
        
        pad_length_input = input_seqs.shape[2]
        n_samples = input_seqs.shape[0]

        pad_length_target = target_seqs.shape[1]

        model_output = model(input_seqs)
        model_output = model_output.permute(1, 0, 2)  # Assuming log probs are computed in network        
        
        n_timesteps = model_output.shape[0]
        input_lengths = torch.tensor([n_timesteps for i in range(n_samples)]).to(device)
        label_lengths = torch.tensor([len(y[ind + i]) for i in range(n_samples)]).to(device)
        
        loss = ctc(
            log_probs=model_output, targets=target_seqs, input_lengths=input_lengths, target_lengths=label_lengths)
        
        
        if train:
            try:
                loss.backward()
            except Exception as e:
                print(f"Exception {e}")
                continue
            optimizer.step()
        
        losses[int(ind / batch_size) - 1] = loss.item()
        model_output = model_output.permute(1, 0, 2).detach().cpu()
        torch.cuda.empty_cache()

        with torch.no_grad():
            # Iterating over batch size to get all sequence predictions
            for k in range(3):
                indices = np.random.permutation(batch_size)
                greedy_result = decoder(model_output[indices[k]])
                greedy_transcript = " ".join(greedy_result)
                actual_transcript = get_actual_transcript(
                    y[ind + indices[k]])
                sorted_greedy = sort_transcript(greedy_transcript)
                sorted_actual = sort_transcript(actual_transcript)
                motifs_found, motif_errs = evaluate_cycle_prediction(
                    sorted_greedy, sorted_actual)
                edit_distance_ratios.append(ratio(
                    greedy_transcript, actual_transcript))
                motifs_found_arr.append(motifs_found)
                motif_errs_arr.append(motif_errs)
        

        #print(f"\n{ratio_labels} aah {len(y[ind])}")
        
        if int(ind / batch_size) % 50 == 0 and not ind == 0:
            print(f"\nGreedy Transcript \n{greedy_transcript}")
            print(f"Actual Transcript \n{actual_transcript}")
            print(f"Cycle Greedy \n{sorted_greedy}")
            print(f"Cycle Actual \n{sorted_actual}")

        
    return {
        "model": model,
        "losses": losses,
        "edit_distance_ratios": edit_distance_ratios,
        "motifs_found": motifs_found_arr,
        "motif_errs": motif_errs_arr
    }
        

def main(
        n_classes: int, epochs: int = 50, sampling_rate: float = 1.0,
        window_size: int = 1024, window_step: int = 800,
        running_on_hpc: bool = False, windows: bool = True,
        dataset_path: str = None, hidden_size: int = 1024, n_layers: int = 3,
        dataset: str = "", normalize_flag: bool = False, lr:int = 0.001, batch_size:int = 1,
        saved_model: bool = False, saved_model_path: str = None):

    if dataset_path:
        _, model_save_path, file_write_path = get_savepaths(
            running_on_hpc=running_on_hpc)

    else:
        dataset_path, model_save_path, file_write_path = get_savepaths(
            running_on_hpc=running_on_hpc)
    
    X, y = load_training_data(
        dataset_path=dataset_path, column_x='squiggle', column_y='edit_spacer_seq',
        sampling_rate=sampling_rate, orientation=False)

    if windows:
        X = data_preproc(
            X=X, window_size=window_size, step_size=window_step,
            normalize_values=normalize_flag)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=1) 
    torch.autograd.set_detect_anomaly(True)

    output_size = n_classes
    dropout_rate = 0.2
    model_save_epochs = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    print(f"Running on {device}")

    if saved_model:
        model = load_model(
            model_path=saved_model_path, device=device, n_classes=n_classes, hidden_size=hidden_size)
        print("\nLoaded saved model\n")
    else:
        model = NaiveCaller(num_classes=n_classes, hidden_dim=hidden_size).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    labels_int = np.arange(n_classes).tolist()
    labels = [f"{i}" for i in labels_int]  # Tokens to be fed into greedy decoder
    greedy_decoder = GreedyCTCDecoder(labels=labels)

    model_config = ModelConfig(
        n_classes=n_classes, hidden_size=hidden_size, window_size=window_size,
        window_step=window_step, train_epochs=epochs, device=device,
        model_save_path = model_save_path, write_path=file_write_path,
        dataset=dataset, windows=windows, sampling_rate=sampling_rate,
        batch_size=batch_size
    )

    print(model_config.__dict__())
    
    ctc = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    if running_on_hpc:
        project_name = "motifcaller_hpc_runs"
    else:
        project_name = "motifcaller_local_runs"

    # Training monitoring on wandb
    wandb_login(running_on_hpc=running_on_hpc)
    start_wandb_run(model_config=model_config, project_name=project_name)

    for epoch in range(epochs):
        
        if batch_size == 1:
            train_dict = run_epoch(
                model=model, model_config=model_config, optimizer=optimizer, decoder=greedy_decoder,
                X=X_train, y=y_train, ctc=ctc, train=True, display=False, windows=windows, normalize_flag=normalize_flag,
                batch_size=batch_size
            )
        else:
            train_dict = run_epoch_batched(
                model=model, model_config=model_config, optimizer=optimizer, decoder=greedy_decoder,
                X=X_train, y=y_train, ctc=ctc, train=True, display=False, windows=windows, normalize_flag=normalize_flag,
                batch_size=batch_size
            )

        model = train_dict['model']
        training_losses = train_dict['losses']
        training_ratios = train_dict['edit_distance_ratios']
        training_motifs_found = train_dict['motifs_found']
        training_motif_errs = train_dict['motif_errs']
        
        print(f"\nTrain Epoch {epoch}\n Mean Loss {np.mean(training_losses)}"
              f"\n Mean ratio {np.mean(training_ratios)}"
              f"\n Mean motifs identified {np.mean(training_motifs_found)}"
              f"\n Mean motif errs {np.mean(training_motif_errs)}")
            
        if batch_size == 1:
            validate_dict = run_epoch(
                model=model, model_config=model_config, optimizer=optimizer, decoder=greedy_decoder,
                X=X_val, y=y_val, ctc=ctc, display=False, windows=windows, normalize_flag=normalize_flag,
                batch_size=batch_size
                )
        else:
            validate_dict = run_epoch_batched(
            model=model, model_config=model_config, optimizer=optimizer, decoder=greedy_decoder,
            X=X_val, y=y_val, ctc=ctc, display=False, windows=windows, normalize_flag=normalize_flag,
            batch_size=batch_size
            )
        
        
        validation_losses = validate_dict['losses']
        validation_ratios = validate_dict['edit_distance_ratios']
        validation_motifs_found = validate_dict['motifs_found']
        validation_motif_errs = validate_dict['motif_errs']

        metrics = {
            "train/accuracy": np.mean(training_ratios),
            "train/loss": np.mean(training_losses),
            "train/motifs_found": np.mean(training_motifs_found),
            "train/motif_errs": np.mean(training_motif_errs),
            "validation/accuracy": np.mean(validation_ratios),
            "validation/loss": np.mean(validation_losses),
            "validation/motifs_found": np.mean(validation_motifs_found),
            "validation/motif_errs": np.mean(validation_motif_errs),
        }

        wandb.log(metrics)

        # Schedule learning rate change
        #scheduler.step(np.mean(validation_losses))
        #current_lr = optimizer.param_groups[0]['lr']
        #print(current_lr)

        print(f"\nValidation Epoch {epoch}\n Mean Loss {np.mean(validation_losses)}"
              f"\n Mean ratio {np.mean(validation_ratios)}"
              f"\n Mean motifs identified {np.mean(validation_motifs_found)}"
              f"\n Mean motif errs {np.mean(validation_motif_errs)}")
        #print(random.sample(validation_greedy_transcripts, 1))


        with open(file_write_path, 'a') as f:
            f.write(f"\nEpoch {epoch}\n Training loss {np.mean(training_losses)}\n"
                    f"Validation loss {np.mean(validation_losses)}")
            f.write(f"\nEdit distance ratio: Training {np.mean(training_ratios)}\n"
                    f"Validation {np.mean(validation_ratios)}")
            f.write(f"Motif metrics (found/err): \n{np.mean(training_motifs_found)}{np.mean(training_motif_errs)}\n"
                    f"\n{np.mean(validation_motifs_found)}{np.mean(validation_motif_errs)}\n")
        
        if epoch % model_save_epochs == 0 and epoch > 0:
            if model_save_path:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, model_save_path)
    
    if batch_size == 1:
        test_dict = run_epoch(
            model=model, model_config=model_config, optimizer=optimizer, decoder=greedy_decoder,
            X=X_test, y=y_test, ctc=ctc, windows=windows, normalize_flag=normalize_flag,
            batch_size=batch_size
            )
    else:
        test_dict = run_epoch_batched(
            model=model, model_config=model_config, optimizer=optimizer, decoder=greedy_decoder,
            X=X_test, y=y_test, ctc=ctc, windows=windows, normalize_flag=normalize_flag,
            batch_size=batch_size
            )
    
    test_losses = test_dict['losses']
    test_ratios = test_dict['edit_distance_ratios']
    test_motifs_found = test_dict['motifs_found']
    test_motif_errs = test_dict['motif_errs']
    print(f"\nTest Loop\n Mean Loss {np.mean(test_losses)}\n"
          f"Mean ratio {np.mean(test_ratios)}"
          f"\n Mean motifs identified {np.mean(test_motifs_found)}"
          f"\n Mean motif errs {np.mean(test_motif_errs)}")

    with open(file_write_path, 'a') as f:
        f.write(f"\nTest loss {np.mean(test_losses)}\n Test ratio"
                f" {np.mean(test_ratios)}\nTest motifs :\n"
                f"{test_motifs_found}{test_motif_errs}")
        
    metrics = {
            "test/accuracy": np.mean(test_ratios),
            "test/loss": np.mean(test_losses),
            "test/motifs_found": np.mean(test_motifs_found),
            "test/motif_errs": np.mean(test_motif_errs)
        }

    wandb.log(metrics)

    if model_save_path:
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, model_save_path)