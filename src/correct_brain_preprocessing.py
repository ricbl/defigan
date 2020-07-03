"""Skull Striping part 2

Does skull stripping to brain volumes using BET. This is applied to the ADNI 
dataset after the use of preprocess_adni_all.py

by Ricardo Bigolin Lanfredi
Last modified: 2020-07-01
Project: DeFI-GAN
GNU General Public License v3.0
"""

from subprocess import Popen
import glob
import pathlib
import os
n_processes = 63
processes = []

bmicdatasets_root = '~/ADNI/Processed_part_2/'

#expects all the volumes to reprocess to be in a folder named Processed_part_2/ADNI_all_no_PP_3/rid_<id>,
#where id is the patient ADNI id.
# it writes the processed files in the same structure, but on folder Processed_part_3 instead of Processed_part_2
def restrip_skull(origin_folder):
    for index_file, file in enumerate(glob.glob(origin_folder + "/**/*.nii.gz", recursive=True)):
        tmp_file_path2 = file.replace('/Processed_part_2/', '/Processed_part_3/')
        assert(file!=tmp_file_path2)
        pathlib.Path(os.path.dirname(tmp_file_path2)).mkdir(parents=True, exist_ok=True)
        processes.append(Popen('bet {0} {1} -R -f 0.25 -g 0'.format(file, tmp_file_path2), shell=True))
        if len(processes)>n_processes:
            processes[index_file-n_processes].communicate()
            print("Process " + str(index_file-n_processes) + " completed")

def main():
    restrip_skull(bmicdatasets_root)

if __name__ == '__main__':
    main()