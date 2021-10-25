import datetime

import torch
import dataset

from os import path, makedirs


def create_model(num_classes):
    pass


def main(parser_data):
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    args = parser.parse_args()
    print(f'The args is: {args}')

    if not path.exists(args.output_dir):
        makedirs(args.output_dir)

    main(args)