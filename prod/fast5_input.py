
from ont_fast5_api.fast5_interface import get_fast5_file
import os
from tqdm import tqdm


def extract_fast5_data_from_file(fast5_filepath):
    squiggles = []
    with get_fast5_file(fast5_filepath, mode="r") as f5:
        read_ids = f5.get_read_ids()
        squiggles = [f5.get_read(read_id).get_raw_data() for read_id in read_ids]

    return squiggles, read_ids

"""
Down the line
def extract_fast5_data_from_folder(fast5_path):

    total_files = len(os.listdir(fast5_path))
    squiggles = {}

    for file in tqdm(os.listdir(fast5_path), total=total_files):

        read_ids_arr = []
        squiggles_arr = []
        filepath = os.path.join(fast5_path, file)
        with get_fast5_file(filepath, mode="r") as f5:

            read_ids = f5.get_read_ids()

            for read_id in read_ids:
                read = f5.get_read(read_id)
                squiggle = read.get_raw_data()
                squiggles[read_id] = squiggle

    return squiggles
"""