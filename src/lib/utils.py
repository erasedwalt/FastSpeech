import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio

import math


SPEC_FILL = math.log(1e-5)


def rescale_pivots(pivots_wave, spec_len):
    '''
    Rescale alignment pivots from waveform domain to
    spectrogram domain

    Args:
        pivots_wave (Tensor):
            Alignment pivots in waveform domain
            (cumulative sum of Aligner scaled outputs)
        spec_len (Tensor):
            Spectogram length
    '''
    wave_len = pivots_wave[-1]
    coeff = spec_len / wave_len
    pivots_spec = coeff * pivots_wave
    pivots_spec[-1] = spec_len
    pivots_spec = pivots_spec.int()
    durations = pivots_spec - torch.cat([torch.tensor([0]), pivots_spec[:-1]])
    return durations


def create_mask(src_tensor, lengths):
    '''
    Create mask

    Args:
        src_tensor (Tensor):
            Source tensor
        lengths (Tensor):
            Sequence lengths
    '''
    return (torch.arange(src_tensor.shape[-1], device=lengths.device)[None, :] \
            < lengths[:, None])


def rescale_durations(durations):
    '''
    Rescale the lengths so that their sum is equal to 1
    
    Args:
        durations (Tensor):
            Tensor with durations
    '''
    durations = (1. / durations.sum()) * durations
    return durations


def calc_alignment_and_spec(melspec, aligner, waveform, waveform_length, token_length, transcript):
    '''
    Calculate alignments (teacher phoneme durations) and
    calculate spectrograms

    Args:
        aligner (nn.Module):
            Teacher aligner
        waveform (Tensor):
            Waveforms tensor
        waveform_length (Tensor):
            Waveforms lengths
        token_length (Tensor):
            Token length
        transcript (Tuple):
            Waveform transcripts
    '''
    with torch.no_grad():
        teacher_durations_raw = aligner(waveform, waveform_length, transcript)

    teacher_durations = []
    specs = []
    spec_length = []
    for idx in range(waveform.shape[0]):
        durations_ = teacher_durations_raw[idx][:token_length[idx]]
        durations_ = rescale_durations(durations_)
        pivots_wave = torch.round((durations_ * waveform_length[idx]).cumsum(dim=0)).int()

        spec = melspec(waveform[idx][:pivots_wave[-1]])
        specs.append(spec.squeeze().transpose(0, 1))
        spec_length.append(spec.shape[-1])
        durations_ = rescale_pivots(pivots_wave, spec.shape[-1])
        teacher_durations.append(durations_)

    specs = pad_sequence(specs, batch_first=True, padding_value=SPEC_FILL).transpose(1, 2)
    spec_length = torch.tensor(spec_length)
    teacher_durations = pad_sequence(teacher_durations, batch_first=True).int()
    return specs, spec_length, teacher_durations
