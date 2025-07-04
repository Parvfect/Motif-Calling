
import argparse
import os 
from datetime import datetime
from inference import model_init, model_inference
from output import save_inference_to_csv
import os

parser = argparse.ArgumentParser(
                    prog='Motif Caller',
                    description='Inference for the motif caller')

parser.add_argument('--fast5_path', type=str)
parser.add_argument('--savepath', type=str)
parser.add_argument('--fast5_directory', type=str)

parser.set_defaults(
    fast5_path="",
    savepath=""
    )

args = parser.parse_args()

if __name__ == '__main__':
    fast5_path = args.fast5_path
    savepath = args.savepath
    fast5_directory = args.fast5_directory

    if not savepath:
        print("No savepath provided, saving output to current directory\n")

    if fast5_path:
        print("Initialising model and reading data\n")
        squiggles, read_ids, forward_model, mixed_model, reverse_model, device, greedy_decoder = model_init(fast5_path=fast5_path)
        print("Starting inference\n")
        sorted_greedy_transcripts, greedy_transcripts_arr, read_ids_arr, qualities, full_qs = model_inference(
            data_arr=squiggles, read_ids=read_ids, forward_model=forward_model, mixed_model=mixed_model, reverse_model=reverse_model, device=device, greedy_decoder=greedy_decoder
        )
        print("Saving results\n")
        save_inference_to_csv(sorted_greedy_transcripts=sorted_greedy_transcripts, greedy_transcripts_arr=greedy_transcripts_arr, read_ids_arr=read_ids_arr, qualities=qualities, full_qs=full_qs, savepath=savepath, fast5_filepath=fast5_path)

    elif fast5_directory:

        for file in os.listdir(fast5_directory):
            if file.endswith('fast5'):
                print(f"Loading data from {file}\n")

                print("Initialising model and reading data\n")
                squiggles, read_ids, forward_model, mixed_model, reverse_model, device, greedy_decoder = model_init(fast5_path=os.path.join(fast5_directory, file))

                print("Starting inference\n")
                sorted_greedy_transcripts, greedy_transcripts_arr, read_ids_arr, qualities, full_qs = model_inference(
                    data_arr=squiggles, read_ids=read_ids, forward_model=forward_model, mixed_model=mixed_model, reverse_model=reverse_model, device=device, greedy_decoder=greedy_decoder
                )
                print("Saving results\n")
                save_inference_to_csv(sorted_greedy_transcripts=sorted_greedy_transcripts, greedy_transcripts_arr=greedy_transcripts_arr, read_ids_arr=read_ids_arr, qualities=qualities, full_qs=full_qs, savepath=savepath, fast5_filepath=file)

    else:
        print("No fast5 path provided!")
        exit()