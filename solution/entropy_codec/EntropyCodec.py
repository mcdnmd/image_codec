import imp
import pkg_resources


def __bootstrap__():
    global __bootstrap__, __loader__, __file__

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
