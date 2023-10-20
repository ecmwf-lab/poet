# Copyright (c) Microsoft Corporation and the European Centre for Medium-Range Weather Forecasts.
# Licensed under the MIT License.

import argparse
import os
from pathlib import Path
import yaml


def get_file_name(var, lead, pre, suf, ext):
    return f"{pre}{var}_{lead}h{suf}.{ext}"


def create_config(args):
    if os.path.isfile(args.output_file):
        raise IOError(f"output file {args.output_file} exists")

    os.makedirs(Path(args.output_file).parent, exist_ok=True)

    config = dict()
    config['env'] = {
        'bucket': args.bucket,
        'scratch_directory': args.scratch_directory
    }

    config['files'] = list()
    for var in args.variables:
        for lead in args.lead_times:
            config['files'].append(get_file_name(var, lead, args.prefix, args.suffix, args.extension))

    with open(args.output_file, 'w') as f:
        yaml.dump(config, f)
        print(f"wrote config to {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, required=True,
                        help="Output file to write config")
    parser.add_argument('--variables', type=lambda v: v.split(','), required=True,
                        help="Comma-separated list of string variable names")
    parser.add_argument('--lead_times', type=lambda ti: [int(t) for t in ti.split(',')], required=True,
                        help="Comma-separated list of integer lead times in hours")
    parser.add_argument('--bucket', type=str, default='s3://poet_dataset',
                        help="Config: path to data bucket")
    parser.add_argument('--scratch_directory', type=str, default='/mnt/scratch',
                        help="Config: path to scratch working space")
    parser.add_argument('--prefix', type=str, default='',
                        help="File name prefix")
    parser.add_argument('--suffix', type=str, default='',
                        help="File name suffix")
    parser.add_argument('--extension', type=str, default='zarr',
                        help="File extension")

    run_args = parser.parse_args()
    create_config(run_args)
