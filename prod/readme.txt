----Motif Caller: Inference-----

Running instructions:

1. Navigate to the directory
2. >> pip install -r requirements.txt
3. >> python process.py --fast5_path <fast5_path> --savepath <savepath>

For now, the model needs to be run on one fast5 file at a time. So the fast5_path should be the whole filepath. 

Running on GPU:
To run on the GPU, pytorch needs to be correctly installed to utilise CUDA. The code will automatically run on the GPU if its done so. Pytorch installation instructions are in - https://pytorch.org/get-started/locally/. 

Output type:
Example output file is included in the directory.