import numpy as np


def block_conv(weight, *, output_lanes, spatial_alignment):
    output_channels, input_channels, height, width = weight.shape
    panels = (output_channels + output_lanes - 1) // output_lanes
    spatial = (
        (height * width + spatial_alignment - 1)
        // spatial_alignment
        * spatial_alignment
    )
    padded = np.pad(
        weight,
        ((0, panels * output_lanes - output_channels), (0, 0), (0, 0), (0, 0)),
    )
    packed = padded.reshape(
        panels, output_lanes, input_channels, height, width
    ).transpose(0, 2, 3, 4, 1)
    result = np.zeros(
        (panels, input_channels, spatial, output_lanes), dtype=weight.dtype
    )
    result[:, :, : height * width, :] = packed.reshape(
        panels, input_channels, height * width, output_lanes
    )
    return result
