from pathlib import Path
import re
from datetime import datetime


def inc_build(file_name):
    with open(file_name) as handle:
        for line in handle:
            if line.startswith('__version__'):
                line = line.replace(' ','').replace('__version__=','').replace("'","").strip()
                versions = line.split('.')
                major = versions[0]
                minor = versions[1]
                minor_2 = versions[2]
                new_build = datetime.today().strftime('%d%m%y_%H%M')
                new_version = f"{major}.{minor}.{minor_2}.b{new_build}"
                break
    return new_version


def embed_version(version_file_path):
    new_version = inc_build(version_file_path)
    with open(version_file_path, "wt") as handle:
        handle.write(f"#!/usr/bin/env python3\n"
                f"# -*- coding: utf-8 -*-\n"\
                f"#\n"\
                f"# __init__.py\n"\
                f"\n"\
                f"# SciML-Bench\n"\
                f"# Copyright © 2021 Scientific Machine Learning Research Group\n"\
                f"# Scientific Computing Department, Rutherford Appleton Laboratory\n"\
                f"# Science and Technology Facilities Council, UK.\n"\
                f"# All rights reserved.\n"\
                f"\n"\
                f"# specify version here; it will picked up by pip\n"\
                f"__version__='{new_version}'\n")
    return new_version