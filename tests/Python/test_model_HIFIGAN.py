# documentation: https://pytorch.org/hub/nvidia_deeplearningexamples_hifigan/

import torch
import warnings
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._functorch.aot_autograd import aot_autograd_decompositions

warnings.filterwarnings("ignore")

device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(f"Using {device} for inference")
fastpitch, generator_train_setup = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub",
    "nvidia_fastpitch",
    force_reload=True,
)
hifigan, vocoder_train_setup, denoiser = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub", "nvidia_hifigan", force_reload=True
)


fastpitch.to(device)
hifigan.to(device)
denoiser.to(device)

tp = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub",
    "nvidia_textprocessing_utils",
    cmudict_path="cmudict-0.7b",
    heteronyms_path="heteronyms",
)

text = "Say this smoothly, to prove you are not a robot."

batches = tp.prepare_input_sequence([text], batch_size=1)

gen_kw = {"pace": 1.0, "speaker": 0, "pitch_tgt": None, "pitch_transform": None}
denoising_strength = 0.005

# for batch in batches:
#     with torch.no_grad():
#         mel, mel_lens, *_ = fastpitch(batch['text'].to(device), **gen_kw)
#         audios = hifigan(mel).float()
#         audios = denoiser(audios.squeeze(1), denoising_strength)
#         audios = audios.squeeze(1) * vocoder_train_setup['max_wav_value']

input = batches[0]["text"].to(device)
output_fastpitch = fastpitch(batch["text"].to(device))
# output_hifigan = hifigan(output_fastpitch).float()
# output_denoiser = denoiser(output_hifigan。squeeze(1), denoising_strength)

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=aot_autograd_decompositions,
    is_inference=True,
)


gm, params = dynamo_compiler.importer(fastpitch, input)
