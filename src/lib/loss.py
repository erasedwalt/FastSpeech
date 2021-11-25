import torch
from torch import nn


class FastSpeechLoss(nn.Module):
    '''
    FastSpeech model loss

    MSE Loss on spectrogram part and
    L1 Loss on durations part

    Args:
        pred_spec (Tensor):
            Predicted spectrograms
        specs (Tensor):
            Ground truth spectrograms
        spec_mask (Tensor):
            Boolean spectrogram mask
        pred_durations (Tensor):
            Predicted phoneme durations
        teacher_durations (Tensor):
            Teacher model phoneme duration predictions
        duration_mask (Tensor):
            Boolean duration mask
    '''
    def __init__(self):
        super(FastSpeechLoss, self).__init__()

    def forward(self, pred_spec, specs, spec_mask,
                pred_durations, teacher_durations, durations_mask):

        # Spectrogram loss part (MSE)
        loss = torch.mean(((pred_spec - specs) ** 2) \
            .sum(dim=1) * spec_mask)

        # Duration loss part (L1)
        loss += torch.mean(torch.abs(pred_durations - \
            teacher_durations.float().clamp(min=1e-5).log()) * \
                durations_mask)
        return loss
