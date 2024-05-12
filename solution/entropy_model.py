import numpy as np

from solution.entropy_codec.EntropyCodec import HiddenLayersEncoder, HiddenLayersDecoder


def entropy_encoder(image, size_z: int, size_h: int, size_w: int):
    temp = image.astype(np.uint8).copy()

    maxbinsize = (size_h * size_w * size_z)
    bitstream = np.zeros(maxbinsize, np.uint8)
    StreamSize = np.zeros(1, np.int32, 'C')
    HiddenLayersEncoder(temp, size_w, size_h, size_z, bitstream, StreamSize)
    return bitstream[:StreamSize[0]]


def entropy_decoder(bitstream: np.array, size_z: int, size_h: int, size_w: int):
    decoded_data = np.zeros((size_z, size_h, size_w), np.uint8)
    FrameOffset = np.zeros(1, np.int32)
    HiddenLayersDecoder(decoded_data, size_w, size_h, size_z, bitstream, FrameOffset)
    return decoded_data
