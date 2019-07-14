import glob
import os

import click
import h5py
import imageio
import numpy as np

from unet3d.utils import get_logger

@click.command()
@click.argument('folder')
def main(folder, extension='tif', mapping={'raw': 'raw', 'label': 'mask'}):
    logger = get_logger('ConvertToHDF5Dataset')

    parent = os.path.abspath(os.path.join(folder, os.pardir))
    _current, current = os.path.split(folder)
    if len(current) == 0:
        _, current = os.path.split(_current)
    file_path = os.path.join(parent, current+'.h5')

    logger.info("Converting \"{}\"...".format(folder))

    with h5py.File(file_path, 'w') as output_file:
        for key, item in mapping.items():
            logger.info("Processing dataset \"{}\"".format(key))
            
            search_path = os.path.join(folder, item, '*.'+extension)
            file_list = glob.glob(search_path)
            file_list.sort()

            # load the first file as dummy
            image = imageio.volread(file_list[0])
            shape = (len(file_list), ) + image.shape
            logger.info("Dataset {}, {}".format(shape, image.dtype))

            dataset = output_file.create_dataset(key, shape, dtype=image.dtype)
            for i, file_path in enumerate(file_list):
                logger.info("[{}] {}".format(i, file_path))

                image = imageio.volread(file_path)

                # patch our segmentation result
                #   0: background
                #   1: foreground 
                if key == 'label':
                    image += 1

                dataset[i, ...] = image

if __name__ == '__main__':
    main()