from pathlib import Path
import os
import sys


def package_files(dst_folder):

    SCIML_BENCH_ROOT = Path(__file__).parents[2]  

    # Sequence of operations for building a release:
    # 1. Set the version in sciml_bench/__init__.py
    # 2. Set the new version in the main MD file
    # 3. Build Documentation 
    # 4. Copy Documentation


    # 1
    package_path = SCIML_BENCH_ROOT /  'sciml_bench' 
    root_files = ['LICENSE', 'MANIFEST.in', 'README.md', 'RELEASE_NOTES.md', 'requirements.txt', 'setup.py']


    # 2 
    os.system(f'cp -r  {package_path} {dst_folder}')
    for file in root_files:
        src_file = SCIML_BENCH_ROOT /  file
        os.system(f'cp   {src_file} {dst_folder}/')






if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(f'Usage: {sys.argv[0]} DST_FOLDER')
        print(f'\t Where DST_FOLDER is the destination folder')
    else:
        package_files(sys.argv[1])