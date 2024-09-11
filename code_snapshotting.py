# copy code snapshots for each run to the results folder (runs/<exp number>)
# SRA (paper link)
import shutil
import os
import glob

def copy_files_with_timestamp(source_patterns, save_root, run_timestamp=None):
    save_dir = save_root
    os.makedirs(save_dir, exist_ok=True)
    print('taking snapshot of the main codes')

    source_files = []
    for pattern in source_patterns:
        source_files.extend(glob.glob(pattern))

    for src_file in source_files:
        if os.path.isfile(src_file):  # Check if it's a file, not dir
            filename_with_extension = os.path.basename(src_file)
            filename, extension = os.path.splitext(filename_with_extension)
            dst_file = f'{save_dir}/{filename}_{run_timestamp}{extension}'
            shutil.copy(src_file, dst_file)
        else:
            print(f'Skipping directory: {src_file}')