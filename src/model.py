import torch
import torch.nn import nn
import random
import os
import soundfile as sf
import pandas as pd
import numpy as np
from asteroid_filterbanks import make_enc_dec

from .utils import GlobLN, pad_x_to_y
from .separator import DPTransformer
from .modulator import Modulator


class AvaTr(nn.Module):
    """ Avatar based speech separation.

    Args:
        in_chan (int, optional): Number of input channels, should be equal to n_filters.
        n_src (int): Number of masks to estimate.
        ff_hid (int): Number of neurons in the RNNs cell state.
            Defaults to 256.
        chunk_size (int): window size of overlap and add processing.
            Defaults to 100.
        hop_size (int or None): hop size (stride) of overlap and add processing.
            Default to `chunk_size // 2` (50% overlap).
        n_repeats (int): Number of repeats. Defaults to 6.
        norm_type (str, optional): Type of normalization to use. To choose from
            - ``'gLN'``: global Layernorm
            - ``'cLN'``: channelwise Layernorm
        ff_activation (str, optional): activation function applied at the output of RNN.
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        dropout (float, optional): Dropout ratio, must be in [0,1].
        enc_activation (str, optional): activation function applied at the output of encoder.
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``].
        n_filters (int): Number of filters / Input dimension of the masker net.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.
    """

    def __init__(
        self,
        # modulator params
        n_spk,
        embed_dim=512,
        # DPT params
        in_chan=128,
        n_src=2,
        ff_hid=256,
        chunk_size=100,
        hop_size=None,
        n_repeats=6,
        norm_type="gLN",
        ff_activation="relu",
        mask_act="relu",
        bidirectional=True,
        dropout=0,
        # encoder/decoder params
        enc_activation="relu",
        fb_name="free",
        kernel_size=16,
        n_filters=64,
        stride=8,
        **fb_kwargs,
    ):
        self.encoder, self.decoder = make_enc_dec(
            fb_name, kernel_size=kernel_size, n_filters=n_filters, stride=stride, **fb_kwargs
        )

        n_feats = encoder.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )

        self.enc_activation = nn.ReLU() if enc_activation == 'relu' else nn.Identity()
        self.enc_norm = GlobLN(n_feats)

        self.modulator = Modulator(n_spk, in_chan, embed_dim)

        self.masker = DPTransformer(
            n_feats,
            n_src,
            n_heads=4,
            ff_hid=ff_hid,
            chunk_size=chunk_size,
            hop_size=hop_size,
            n_repeats=n_repeats,
            norm_type=norm_type,
            ff_activation=ff_activation,
            mask_act=mask_act,
            bidirectional=bidirectional,
            dropout=dropout,
        )

    def forward(self, inputs):
        """ Enc/Mask/Dec model forward

        Args:
            inputs = (wav, spk_id)
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.
            spk_id (torch.Tensor): speaker id

        Returns:
            torch.Tensor, of shape (batch_size, time).
        """
        wav, spk_id = inputs
        spk_id = spk_id.squeeze(-1) # B x 1 -> B

        # Handle 1D, 2D or n-D inputs
        if wav.ndim == 1:
            wav = wav.unsqueeze(0).unsqueeze(1)
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)

        # Real forward
        mix_rep_0 = self.enc_norm(self.enc_activation(self.encoder(wav))) # B x C x T
        mix_rep_t = self.modulator(mix_rep_0, spk_id) # B x C x T
        est_masks = self.masker(mix_rep_t) # B x C x T
        masked_rep = est_masks * mix_rep_t
        out_wavs = pad_x_to_y(self.decoder(masked_rep), wav)

        return out_wavs.squeeze(1)


if __name__ == '__main__':
    device = torch.device('cuda:3')

    # Model
    model = AvaTr(n_spk=10)
    model = model.to(device)

    mix = torch.rand(2, 24000)
    spk_id = torch.LongTensor([1, 8])
    mix, spk_id = mix.to(device), spk_id.to(device)
    est = model(mix, spk_id)
