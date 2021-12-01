import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import math

from .utils import create_mask


SPEC_FILL = math.log(1e-5)


class MultiHeadAttention(nn.Module):
    '''
    Multi-Head Attention from 
    https://arxiv.org/pdf/1706.03762.pdf

    Expects input with shape (bsz, seq_len, in_size)

    Args:
        in_size (int):
            Input tensor embedding dim
        hidden_size (int):
            Attention hidden size
        num_heads (int):
            Number of heads in attention
        dropout (float):
            Dropout probability
    '''
    def __init__(self, in_size, hidden_size, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()

        self.in_size = torch.tensor(in_size)
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.W_Q = nn.Linear(in_size, hidden_size * num_heads, bias=True)
        self.W_K = nn.Linear(in_size, hidden_size * num_heads, bias=True)
        self.W_V = nn.Linear(in_size, hidden_size * num_heads, bias=True)
        self.W_O = nn.Linear(num_heads * hidden_size, in_size, bias=True)
        self.D = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)

    def forward(self, x, mask=None, attentions=None):
        bsz, length = x.shape[0], x.shape[1]

        # x: (bsz, len, emb_size)
        queries = self.W_Q(x).view(bsz, length, self.num_heads, self.hidden_size).transpose(1, 2)
        keys = self.W_K(x).view(bsz, length, self.num_heads, self.hidden_size).transpose(1, 2)
        values = self.W_V(x).view(bsz, length, self.num_heads, self.hidden_size).transpose(1, 2)

        # q, k, v: (bsz, heads, len, hidden)
        energies = torch.matmul(queries, keys.transpose(-1, -2)) / torch.sqrt(self.in_size)

        # energies: (bsz, heads, len, len)
        # mask: (bsz, 1, 1, len)
        if mask is not None:
            energies = energies.masked_fill(~mask, -1e9) 
        alphas = energies.softmax(dim=-1)
        if attentions is not None:
            attentions.append(alphas.detach().cpu())

        # replace heads <-> len
        z_s = torch.matmul(alphas, values).transpose(1, 2).contiguous().view(bsz, length, -1)
        z = self.W_O(z_s)
        z = self.D(z)
        return z


class PositionalEncoding(nn.Module):
    '''
    Positional Encoding from
    https://arxiv.org/pdf/1706.03762.pdf

    Expects input with shape (seq_len, bsz, in_size)

    Args:
        in_size (int):
            Input tensor embedding dim
        dropout (float):
            Probability of dropout
        max_len (int):
            Input sequence maximum length
    '''
    def __init__(self, in_size: int, dropout, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_size, 2) * (-math.log(10000.0) / in_size))
        encoding = torch.zeros(max_len, 1, in_size)
        encoding[:, 0, 0::2] = torch.sin(position * div_term)
        encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('encoding', encoding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (seq_len, batch_size, embedding_dim)
        x += self.encoding[:x.size(0)]
        x = self.dropout(x)
        return x


class FeedForwardBlock(nn.Module):
    '''
    Feed Forward Transformer Block from 
    https://arxiv.org/pdf/1905.09263.pdf

    Uses PreNorm sequence style from
    https://tunz.kr/post/4

    Args:
        num_heads (int):
            Number of heads
        hidden_size (int):
            Hidden size in Block
        conv_hidden_size (int):
            Hidden size in convolution layers
        kernel_size (int):
            Kernel size in convolution layers
        dropout (float):
            Dropout probability
    '''
    def __init__(self, num_heads, hidden_size,
                 conv_hidden_size, kernel_size,
                 groups, dropout):
        super(FeedForwardBlock, self).__init__()

        self.MHA = MultiHeadAttention(
            in_size=hidden_size, 
            hidden_size=hidden_size, 
            num_heads=num_heads,
            dropout=dropout
        )
        self.N1 = nn.LayerNorm(hidden_size)
        self.C = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=conv_hidden_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=groups
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=conv_hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=groups
            ),
            nn.Dropout(dropout)
        )
        self.N2 = nn.LayerNorm(hidden_size)

        for module in self.C:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x, mask=None, attentions=None):
        # x: (bsz, len, emb_size)
        residual = x
        x = self.N1(x)
        x = self.MHA(x, mask, attentions)

        # x: (bsz, len, emb_size)
        x += residual
        residual = x
        x = self.N2(x)

        if mask is not None:
          x = x.transpose(-1, -2)
          x = x.masked_fill(~(mask.squeeze().unsqueeze(-2)), 0.)
          x = x.transpose(-1, -2)

        # transpose len <-> emb_size for convs
        x = self.C(x.transpose(1, 2)).transpose(1, 2)
        x += residual

        if mask is not None:
          x = x.transpose(-1, -2)
          x = x.masked_fill(~(mask.squeeze().unsqueeze(-2)), 0.)
          x = x.transpose(-1, -2)

        return x


class FeedForwardTransformer(nn.Module):
    '''
    Feed Forward Transformer from
    https://arxiv.org/pdf/1905.09263.pdf

    Args:
        n_repeats (int):
            Number of block repeats
        num_heads (int):
            Number of heads in attention
        hidden_size (int):
            Hidden size
        conv_hidden_size (int):
            Hidden size in convolution layers
        kernel_size (int):
            Kernel size in convolution layers
        groups (int):
            Number of groups in convolution layers
        dropout (float):
            Dropout probability
        max_len (int):
            Maximum input sequence length
    '''
    def __init__(self, n_repeats, num_heads, hidden_size,
                 conv_hidden_size, kernel_size, groups, dropout, max_len):
        super(FeedForwardTransformer, self).__init__()

        self.PE = PositionalEncoding(hidden_size, dropout, max_len)
        self.Bs = nn.ModuleList([
            FeedForwardBlock(
                num_heads=num_heads,
                hidden_size=hidden_size,
                conv_hidden_size=conv_hidden_size,
                kernel_size=kernel_size,
                groups=groups,
                dropout=dropout
            )
            for i in range(n_repeats)
        ])
        self.LN = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None, attentions=None):
        # x: (batch, len, emb_size)
        x = self.PE(x.transpose(0, 1)).transpose(0, 1)
        for block in self.Bs:
            x = block(x, mask, attentions)
        x = self.LN(x)
        return x


class DurationPredictor(nn.Module):
    '''
    Duration Predictor from
    https://arxiv.org/pdf/1905.09263.pdf

    Args:
        in_size (int):
            Input tensor dimension
        hidden_size (int):
            Hidden size
        kernel_size (int):
            Kernel size
        groups (int):
            Number of groups
        dropout (float):
            Dropout probability
    '''
    def __init__(self, in_size, hidden_size, kernel_size, groups, dropout):
        super(DurationPredictor, self).__init__()

        self.Ms = nn.ModuleList([
            nn.Conv1d(
                in_channels=in_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=groups
            ),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=groups
            ),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, 1)
        ])

        for module in self.Ms:
            if hasattr(module, 'weight') and module.weight.dim() > 1:
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        # x: (bsz, len, emb_size)
        for i, module in enumerate(self.Ms):
            if i in [0, 4]:
                # transpose for convs
                x = module(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = module(x)
        return x.squeeze(2)



class FastSpeech(nn.Module):
    '''
    FastSpeech architechture from
    https://arxiv.org/pdf/1905.09263.pdf

    I changed last layer because outputs
    were ragged and abrupt

    Args:
        vocabulary_size (int):
            Vocabulary size
        num_phoneme_block (int):
            Number of blocks in phoneme part of model
        num_melspec_block (int):
            Number of blocks in spectrogram part of model
        num_heads (int):
            Number of heads in attention
        hidden_size (int):
            Hidden size
        conv_hidden_size (int):
            Hidden size in convolution layers
        kernel_size (int):
            Kernel size in convolution layers
        groups (int):
            Number of groups in convolution layers
        dropout (float):
            Dropout probability
        max_phoneme_len (int):
            Maximum length of phoneme sequence
        max_melspec_len (int):
            Maximum length of spectrogram sequence
        last_conv (bool):
            Whether to use convolution as last layer
    '''
    def __init__(self, vocabulary_size, num_phoneme_block=6, num_melspec_block=6,
                 num_heads=2, hidden_size=384, conv_hidden_size=1536,
                 kernel_size=3, groups=1, dropout=0.1, max_phoneme_len=3000,
                 max_melspec_len=3000, last_conv=False):

        super(FastSpeech, self).__init__()
        self.last_conv = last_conv

        self.PH_E = nn.Embedding(vocabulary_size, hidden_size)

        self.PH_FFT = FeedForwardTransformer(
            n_repeats=num_phoneme_block,
            num_heads=num_heads,
            hidden_size=hidden_size,
            conv_hidden_size=conv_hidden_size,
            kernel_size=kernel_size,
            groups=groups,
            dropout=dropout,
            max_len=max_phoneme_len
        )

        self.DP = DurationPredictor(
            in_size=hidden_size,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            groups=groups,
            dropout=dropout
        )

        self.MS_FFT = FeedForwardTransformer(
            n_repeats=num_melspec_block,
            num_heads=num_heads,
            hidden_size=hidden_size,
            conv_hidden_size=conv_hidden_size,
            kernel_size=kernel_size,
            groups=groups,
            dropout=dropout,
            max_len=max_melspec_len
        )

        if self.last_conv:
            self.L = nn.ModuleList([
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.LayerNorm(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=80,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
            ])
        else:
            self.L = nn.Linear(hidden_size, 80)

        nn.init.normal_(self.PH_E.weight, mean=0., std=1 / math.sqrt(hidden_size))

        if self.last_conv:
            for module in self.L:
                if hasattr(module, 'weight') and module.weight.dim() > 1:
                    nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.xavier_uniform_(self.L.weight)

    def length_regulator(self, x, durations, alpha=1.):
        # x: (bsz, len, emb_size)
        # durations: (bsz, len)
        durations = (durations * alpha).int()

        # expand seq len
        batch = []
        for batch_elem_idx in range(x.shape[0]):
            batch_elem = []
            for frame_idx in range(x.shape[1]):
                one_phoneme = [x[batch_elem_idx, frame_idx, :].unsqueeze(0)
                               for _ in range(durations[batch_elem_idx, frame_idx])]
                batch_elem += one_phoneme
            # batch_elem: (len, emb_size)
            batch_elem = torch.cat(batch_elem, dim=0)
            batch.append(batch_elem)
        melspec_x = pad_sequence(batch).transpose(0, 1) 
        return melspec_x

    def forward(self, x, teacher_durations, ph_mask=None, spec_mask=None):
        # x: (bsz, len)
        x = self.PH_E(x)

        # x: (bsz, len, emb_size)
        # Phoneme part
        x = self.PH_FFT(x, ph_mask)
        durations = self.DP(x)
        x = self.length_regulator(x, teacher_durations)

        # Spectrogram part
        x = self.MS_FFT(x, spec_mask)

        if self.last_conv:
            for i, module in enumerate(self.L):
                if i in [0, 3]:
                    x = module(x.transpose(1, 2)).transpose(1, 2)
                else:
                    x = module(x)
        else:
            x = self.L(x)

        x = x.transpose(1, 2)
        if spec_mask is not None:
            x = x.masked_fill(~spec_mask.squeeze()[:, None, :], SPEC_FILL)
        return x, durations

    def inference(self, x, tokens_length, alpha=1.):
        ph_mask = create_mask(x, tokens_length).to(x.device)
        attentions = []

        # x: (bsz, len)
        x = self.PH_E(x)

        # x: (bsz, len, emb_size)
        x = self.PH_FFT(x, ph_mask[:, None, None, :], attentions)
        durations = torch.exp(self.DP(x))

        # durations: (bsz, len)
        durations = [durations[i][:tokens_length[i]] for i in range(durations.shape[0])]
        durations = torch.round(pad_sequence(durations, batch_first=True)).int()
        spec_length = durations.sum(dim=1)
        x = self.length_regulator(x, durations, alpha)
        spec_mask = create_mask(x.transpose(1, 2), spec_length).to(x.device)
        x = self.MS_FFT(x, spec_mask[:, None, None, :], attentions)

        if self.last_conv:
            for i, module in enumerate(self.L):
                if i in [0, 3]:
                    x = module(x.transpose(1, 2)).transpose(1, 2)
                else:
                    x = module(x)
        else:
            x = self.L(x)

        x = x.transpose(1, 2)
        x = x.masked_fill(~spec_mask[:, None, :], SPEC_FILL)
        return x, durations, attentions
