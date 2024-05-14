import imp
import os

import pkg_resources

from solution import config


def find_file_starting_with(prefix, directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(prefix):
                return os.path.join(root, file)
    return None


def __bootstrap__():
    global __bootstrap__, __loader__, __file__

    path = find_file_starting_with("EntropyCodec", config)

    __file__ = pkg_resources.resource_filename(
        __file__, "EntropyCodec.cpython-312-darwin.so"
    )
    __loader__ = None
    del __bootstrap__, __loader__
    imp.load_dynamic(__name__, __file__)


__bootstrap__()


def HiddenLayersEncoder(layer1, w1, h1, z1):
    ...


def HiddenLayersDecoder(layer1, w1, h1, z1):
    ...


__all__ = ["HiddenLayersEncoder", "HiddenLayersDecoder"]
