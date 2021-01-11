import torch
import torch.nn as nn
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
        n_filters (int): Number of filters / Input dimension of the separator net.
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
        n_filters=128,
        stride=8,
        sample_rate=8000,
        **fb_kwargs,
    ):
        super(AvaTr, self).__init__()

        self.encoder, self.decoder = make_enc_dec(
            fb_name, kernel_size=kernel_size, n_filters=n_filters, stride=stride,
            sample_rate=sample_rate, **fb_kwargs
        )

        n_feats = self.encoder.n_feats_out
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

        self.separator = DPTransformer(
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

    def serialize(self):
        """Serialize model and args

        Returns:
            dict, serialized model with keys `model_args` and `state_dict`.
        """
        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
            model_args=self.get_model_args(),
        )
        return model_conf

    def get_state_dict(self):
        """ In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_model_args(self):
        """return args to re-instantiate the class."""
        fb_config = self.encoder.filterbank.get_config()
        sep_config = self.separator.get_config()
        # Assert both dict are disjoint
        if not all(k not in fb_config for k in sep_config):
            raise AssertionError(
                "Filterbank and Mask network config share common keys. Merging them is not safe."
            )
        # Merge all args under model_args.
        model_args = {
            **fb_config,
            **sep_config,
            "enc_activation": self.enc_activation,
        }
        return model_args

    def forward(self, inputs):
        """
        Args:
            inputs = (wav, spk_id)
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.
            spk_id (torch.Tensor): speaker id

        Returns:
            torch.Tensor, of shape (batch_size, time) or (batch_size, n_src, time).
        """
        wav, spk_id = inputs

        # Handle 1D, 2D or n-D inputs
        if wav.ndim == 1:
            wav = wav.unsqueeze(0).unsqueeze(1)
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)

        # Encoding
        mix_rep_0 = self.enc_norm(self.enc_activation(self.encoder(wav))) # B x C x T
        B, C, T = mix_rep_0.shape

        # Modulation
        mix_rep_t = self.modulator(mix_rep_0, spk_id) # B x n_src x C x T
        mix_rep_t = mix_rep_t.view(-1, C, T) # B * n_src x C x T

        # Masking
        est_masks = self.separator(mix_rep_t) # B * n_src x C x T

        # Decoding
        masked_rep = est_masks.view(B, -1, C, T) * mix_rep_0.unsqueeze(1)
        out_wavs = pad_x_to_y(self.decoder(masked_rep), wav)

        if out_wavs.shape[1] == 1: # task == ehn_single
            out_wavs = out_wavs.squeeze(1)

        return out_wavs
