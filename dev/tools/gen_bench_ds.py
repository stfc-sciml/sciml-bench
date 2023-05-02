#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# gen_bench_ds.py
# Automatically generates the descriptions for the 
# Benchmark and Datasets..

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
Update documentation by registration.yml
WARNING: this script is used by framework maintainers
"""

import yaml
from pathlib import Path

# Converts the YAML file to HTML Table (to avoid MD table issues)
#   context: dataset or benchmark
#   top_key: item from the YAML file 
#   replaced: Changes to the key names
def yaml_to_html_table(context, top_key, replaced):
    """ yaml to html_table """
    # headers
    table = f'<TABLE style="width:100%>"\n'
    table += f'<TR>\n'
    first_column = 'Benchmark' if top_key.lower() == 'benchmarks' else 'Dataset'
    table += f'<TH>{first_column}</TH>\n'
    headers = list(context[top_key].items())[0][1].keys()
    for header in headers:
        if header in replaced.keys():
            header = replaced[header][0]
        table += '<TH>' + header[0].upper() + header[1:] + f' </TH>\n'
    table += f'</TR>\n'
    # cells
    for key, props in context[top_key].items():
        table += f'<TR>\n'
        table += f'<TD>{key}</TD>\n'
        for prop_key, prop_val in props.items():
            if prop_key in replaced.keys():
                prop_val = replaced[prop_key][1](prop_val)
            table += f'<TD>{prop_val}</TD>\n'
        table += '</TR>\n'
    table += f'</TABLE>\n'
    return table

# method to server
def get_server(download_method):
    if download_method == 'STANDARD_DOWNLOAD_STFC_HOST':
        return 'STFC'
    else:
        return 'By contributors'

def build_template():
    # Build the template on the fly
    template  = '# Registered Datasets and Benchmarks'
    template += '\n\n'
    template += '## Benchmarks\n\n'
    template += 'BENCHMARK_TABLE_PLACE_HOLDER'
    template += '\n\n'
    template += 'Please see [CREDITS](./credits.md) for further information.\n'
    template += '\n\n\n\n'
    template += '## Datasets\n\n'
    template += 'DATASET_TABLE_PLACE_HOLDER'
    template += '\n\n'
    template += 'Please see [CREDITS](./credits.md) for further information.\n'
    template += '<div style="text-align: right">◼︎</div>\n\n'

    return template

def main():
    repo_dir = Path(__file__).parents[2]

    # load up the registration.yml
    with open(repo_dir / 'sciml_bench/benchmarks/registration.yml', 'r') as f:
        reg = yaml.load(f, yaml.SafeLoader)

    template = build_template()

    # extract benchmark and dataset descriptions 
    # from the registration file
    dataset_table = yaml_to_html_table(reg, 'datasets',
                              {'size': ('Size (approx)', lambda x: x),
                               'download_method': ('data server', get_server)})
    benchmark_table = yaml_to_html_table(reg, 'benchmarks', {})

    # Now embed these tables into the template
    template = template.replace('BENCHMARK_TABLE_PLACE_HOLDER', benchmark_table)
    template = template.replace('DATASET_TABLE_PLACE_HOLDER', dataset_table)

    # Write Benchmarks and Datasets to a separate file
    with open(repo_dir / 'doc/benchmarks_datasets.md', 'w') as f:
        f.write(template)




if __name__ == "__main__":
    main()