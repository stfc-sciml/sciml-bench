from sciml_bench.core.utils import list_files
from pathlib import Path

path1 = Path("/home/lkt24131/sciml_bench/datasets/em_graphene_sim/inference/raw").expanduser()
path2 = Path("/home/lkt24131/sciml_bench/datasets/em_graphene_sim/inference/truth").expanduser()
inference_file_names =  list_files(path1 , recursive=False)
inference_gt_file_names =  list_files(path2, recursive=False)
for i in range (len(inference_file_names)):
        print (f'{inference_file_names[i]}\t{inference_gt_file_names[i]}')
