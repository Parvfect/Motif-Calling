# Recreate as the model config file
import argparse
import os 
from datetime import datetime
from training_loop import main


parser = argparse.ArgumentParser(
                    prog='Motif Caller',
                    description='Training the motif caller',
                    epilog='Use parser to set some parameters for training')

parser.add_argument('--epochs', type=int, help="Training epochs")
parser.add_argument('--sampling_rate', type=float, help="Sample the data")
parser.add_argument(
    '--no_windows', action='store_true', help="Run model on the whole input sample")
parser.add_argument('--window_size', type=int, help="Window size for ctc")
parser.add_argument('--window_step', type=int, help="Step size for window")
parser.add_argument(
    '--running_on_hpc', action='store_true', help="Flag for running on hpc")
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--n_classes', type=int)
parser.add_argument('--dataset', type=str)
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--lr', type=float)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--saved_model', action='store_true', help='Load from saved model')
parser.add_argument('--saved_model_path', type=str, help="Saved model path")


parser.set_defaults(
    epochs=75, window_size=1024, window_step=800, sampling_rate=1.0,
    running_on_hpc=False, no_windows=False, dataset_path=None, hidden_size=128,
    n_classes=17, dataset="", normalize=False, lr=0.0001, batch_size=1
    )

args = parser.parse_args()

if __name__ == '__main__':
    epochs = args.epochs
    sampling_rate = args.sampling_rate
    window_size = args.window_size
    window_step = args.window_step
    running_on_hpc = args.running_on_hpc
    windows = not args.no_windows
    dataset_path = args.dataset_path
    hidden_size = args.hidden_size
    n_classes = args.n_classes
    dataset = args.dataset
    normalize = args.normalize
    lr = args.lr
    batch_size = args.batch_size
    saved_model = args.saved_model
    saved_model_path = args.saved_model_path

    main(
    n_classes=n_classes, hidden_size=hidden_size, dataset=dataset,
    epochs=epochs, sampling_rate=sampling_rate, window_size=window_size,
    window_step=window_step, running_on_hpc=running_on_hpc, windows=windows,
    dataset_path=dataset_path, normalize_flag=normalize, lr=lr, batch_size=batch_size,
    saved_model=saved_model, saved_model_path=saved_model_path)
