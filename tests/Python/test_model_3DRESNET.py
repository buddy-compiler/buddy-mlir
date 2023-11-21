# RUN: %PYTHON %s 2>&1 | FileCheck %s
# documentation: https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._functorch.aot_autograd import aot_autograd_decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
import urllib.request
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)


def prepare_input():
    device = "cpu"
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 8
    sampling_rate = 8
    frames_per_second = 30

    # Note that this transform is specific to the slow_R50 model.
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size=(crop_size, crop_size)),
            ]
        ),
    )

    # The duration of the input clip is also specific to the model.
    clip_duration = (num_frames * sampling_rate) / frames_per_second
    url_link = (
        "https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4"
    )
    video_path = "archery.mp4"

    urllib.request.urlretrieve(url_link, video_path)

    # The start_sec should correspond to where the action occurs in the video
    start_sec = 0
    end_sec = start_sec + clip_duration

    # Initialize an EncodedVideo helper class and load the video
    video = EncodedVideo.from_path(video_path)

    # Load the desired clip
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

    # Apply a transform to normalize the video input
    video_data = transform(video_data)

    # Move the inputs to the desired device
    inputs = video_data["video"]
    inputs = inputs.to(device)
    return inputs[None, ...]


# model = torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=True)
model = torch.hub.load(
    "facebookresearch/pytorchvideo", "slow_r50", pretrained=True
)


input = prepare_input()
print(input)

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=aot_autograd_decompositions,
    is_inference=True,
)
# normal output test
output = model(input)

gm, params = dynamo_compiler.importer(model, input)
