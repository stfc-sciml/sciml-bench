from pathlib import Path
from embed_version import embed_version
import os
import re



def update_md_file_version(file_path, new_version):
    with open(file_path, 'r') as file:
        text = file.read()

    str_to_find_1= r'Version: \*\*[0-9]\.[0-9].[0-9].b[0-9]*\_[0-9]*\.\*\*'
    str_to_replace_1 = f'Version: **{new_version}.**'

    str_to_find_2= r'Version [0-9]\.[0-9].[0-9].b[0-9]*\_[0-9]*'
    str_to_replace_2 = f'Version {new_version}'

    new_text = re.sub(str_to_find_1, str_to_replace_1, text)
    new_text = re.sub(str_to_find_2, str_to_replace_2, new_text)
    with open(file_path, 'w') as file:
        file.write(new_text)


SCIML_BENCH_ROOT = Path(__file__).parents[2]  

# Sequence of operations for building a release:
# 1. Set the version in sciml_bench/__init__.py
# 2. Set the new version in the main MD file
# 3. Build Documentation 
# 4. Copy Documentation


# 1
version_file_path = SCIML_BENCH_ROOT /  'sciml_bench' / '__init__.py'
new_version = embed_version(version_file_path)


# 2 
main_md_file = SCIML_BENCH_ROOT / 'dev/mkdocs/docs/main/intro.md'
index_md_file = SCIML_BENCH_ROOT / 'dev/mkdocs/docs/index.md'
update_md_file_version(main_md_file, new_version)
update_md_file_version(index_md_file, new_version)

# 3 
dev_docs_path = SCIML_BENCH_ROOT / 'dev/mkdocs' 
doc_build_command = f'cd  {dev_docs_path}; mkdocs build'
os.system(doc_build_command)

# 4
src_files = f'site/*'
tgt_loc_full = SCIML_BENCH_ROOT /  'sciml_bench' / 'docs/full/' 
tgt_loc_bench = SCIML_BENCH_ROOT /  'sciml_bench' / 'docs/benchmarks/' 
tgt_loc_data = SCIML_BENCH_ROOT /  'sciml_bench' / 'docs/datasets/' 

clean_command_full = f'cd {tgt_loc_full}; rm -rf *'
clean_command_bench = f'cd {tgt_loc_bench}; rm -f *'
clean_command_data = f'cd {tgt_loc_data}; rm -f *'

cp_command_full = f'cd {dev_docs_path}; cp -r site/*  {tgt_loc_full}'
cp_command_bench = f'cd {dev_docs_path}; cp -r docs/benchmarks/*  {tgt_loc_bench}'
cp_command_data = f'cd {dev_docs_path}; cp -r docs/datasets/*  {tgt_loc_data}'

os.system(clean_command_full)
os.system(clean_command_bench)
os.system(clean_command_data)

os.system(cp_command_full)
os.system(cp_command_bench)
os.system(cp_command_data)
