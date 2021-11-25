import torch
from torch.nn.utils.rnn import pad_sequence

from typing import Tuple, Dict, List


class LJSpeechCollator:

    def __call__(self, instances: List[Tuple]) -> Dict:
        waveform, waveform_length, tokens, token_length, transcript = list(
            zip(*instances)
        )

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_length = torch.cat(token_length)

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveform_length)

        return waveform, waveform_length, tokens, token_length, transcript
