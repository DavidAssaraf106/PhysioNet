import numpy as np
import random
import os
from scipy.io import loadmat

# Insert the path to your data :Databases A and B in the same folder, with the new Databases (updated SNOMED-CT codes)
input_directory = '/home/david/Training_WFDB_new'


def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file, 'r') as f:
        header_data = f.readlines()
    return data, header_data


if __name__ == '__main__':
    random.seed(30)
    # the scored pathologies sufficiently represented in A and B
    test_pathologies = [270492004, 164889003, 164890007, 713427006, 713426002,
                        164909002, 284470004, 427172004, 59118001, 426177001, 426783006, 427084000, 63593006]
    input_files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith(
                'mat'):
            input_files.append(f)
    input_files.sort()
    print(len(input_files))
    used_files = []
    for i, f in enumerate(input_files):
        tmp_input_file = os.path.join(input_directory, f)
        data, header_data = load_challenge_data(tmp_input_file)
        for iline in header_data:
            if iline.startswith('#Dx'):
                label = iline.split(': ')[1].split(',')[0].strip()  # single label classification
        if test_pathologies.__contains__(int(label)):
            used_files.append(f)
    print(len(used_files))
    training = random.sample(used_files, k=int(0.8 * len(used_files)))
    testing = [file for i, file in enumerate(used_files) if file not in training]
