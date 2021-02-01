import copy
from typing import Optional, List
from dotmap import DotMap

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from asteroid_filterbanks import make_enc_dec
from .utils import GlobLN, pad_x_to_y, DualPathProcessing
from .separator import ImprovedTransformedLayer


class AvaTr(nn.Module):
    def __init__(
        self,
        # Avatars
        n_spk,
        embed_dim=128,
        # Transformer
        d_model=128,
        nhead=4,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        # waveform encoder/decoder params
        kernel_size=16,
        n_filters=128,
        stride=8,
        enc_activation="relu",
        fb_name="free",
        sample_rate=8000,
        **fb_kwargs):

        super().__init__()

        self.model_args = DotMap()
        # Avatars
        self.model_args.n_spk = n_spk
        self.model_args.embed_dim = embed_dim
        # Transformer
        self.model_args.d_model = d_model
        self.model_args.nhead = nhead
        self.model_args.num_encoder_layers = num_encoder_layers
        self.model_args.num_decoder_layers = num_decoder_layers
        self.model_args.dim_feedforward = dim_feedforward
        self.model_args.dropout = dropout
        self.model_args.activation = activation
        self.model_args.normalize_before = normalize_before
        self.model_args.return_intermediate_dec = return_intermediate_dec
        # waveform encoder/decoder params
        self.model_args.kernel_size = kernel_size
        self.model_args.n_filters = n_filters
        self.model_args.stride = stride
        self.model_args.fb_name = fb_name
        self.model_args.sample_rate = sample_rate

        # conv & deconv
        self.conv, self.deconv = make_enc_dec(
            fb_name, kernel_size=kernel_size, n_filters=n_filters, stride=stride,
            sample_rate=sample_rate, **fb_kwargs
        )

        n_feats = self.conv.n_feats_out
        assert d_model == n_feats, (
            "Number of filterbank output channels"
            " and number of model channels should "
            "be the same. Received "
            f"{n_feats} and {d_model}"
        )

        self.enc_activation = nn.ReLU() if enc_activation == 'relu' else nn.Identity()
        #self.enc_norm = GlobLN(n_feats)

        # Avatars
        self.avatar = nn.Embedding(n_spk, embed_dim)

        # Decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        # Encoder
        self.encoder = DPTEncoder(
            d_model,
            n_heads=nhead,
            ff_hid=256,
            chunk_size=100,
            hop_size=None,
            n_repeats=num_encoder_layers,
            norm_type="gLN",
            ff_activation=activation,
            bidirectional=True,
            dropout=dropout,
        )

        self.mask_act = nn.Sigmoid()

        #self.pos_conv = nn.Conv1d(
        #    d_model, d_model,
        #    kernel_size=129,
        #    padding=128 // 2,
        #    groups=16,
        #)

    def forward(self, inputs, mask=None):
        wav, spk_id = inputs

        # Handle 1D, 2D or n-D inputs
        if wav.ndim == 1:
            wav = wav.unsqueeze(0).unsqueeze(1)
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)

        # mix feature
        #mix_rep_0 = self.enc_norm(self.enc_activation(self.conv(wav))) # B x C x T
        mix_rep_0 = self.enc_activation(self.conv(wav)) # B x C x T

        # DPT encoding
        mix_rep_t = self.encoder(mix_rep_0) # B x C x T

        # positional encoding
        pos_embed = None #self.pos_conv(mix_rep_0) # B x C x T

        # avatar embedding TODO: B * n_src x C
        query_embed = self.avatar(spk_id) # B x C
        query_embed = query_embed.unsqueeze(0).repeat(T, 1, 1) # B x C -> T x B x C

        # mask generation
        tgt = torch.zeros_like(query_embed)
        memory = mix_rep_t.permute(2, 0, 1) # T, B, C
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed) # T, B, C
        hs = hs.permute(1, 2, 0)
        mix_mask = self.mask_act(hs)

        # masking
        masked_rep = mix_rep_t * mix_mask

        # source prediction
        out_wavs = pad_x_to_y(self.deconv(masked_rep), wav)

        if out_wavs.shape[1] == 1: # task == ehn_single
            out_wavs = out_wavs.squeeze(1)

        return out_wavs

    def serialize(self, scheduler_dict):
        """Serialize model and args

        Returns:
            dict, serialized model with keys `model_args` and `state_dict`.
        """
        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
            model_args=self.get_model_args(),
            sched_dict=scheduler_dict
        )
        return model_conf

    def get_state_dict(self):
        """ In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_model_args(self):
        """return args to re-instantiate the class."""
        return self.model_args.toDict()


class DPTEncoder(nn.Module):
    """Dual-path Transformer introduced in [1].

    Args:
        in_chan (int): Number of input filters.
        n_heads (int): Number of attention heads.
        ff_hid (int): Number of neurons in the RNNs cell state.
            Defaults to 256.
        chunk_size (int): window size of overlap and add processing.
            Defaults to 100.
        hop_size (int or None): hop size (stride) of overlap and add processing.
            Default to `chunk_size // 2` (50% overlap).
        n_repeats (int): Number of repeats. Defaults to 6.
        norm_type (str, optional): Type of normalization to use.
        ff_activation (str, optional): activation function applied at the output of RNN.
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        dropout (float, optional): Dropout ratio, must be in [0,1].

    References
        [1] Chen, Jingjing, Qirong Mao, and Dong Liu. "Dual-Path Transformer
        Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation."
        arXiv (2020).
    """

    def __init__(
        self,
        in_chan,
        n_heads=4,
        ff_hid=256,
        chunk_size=100,
        hop_size=None,
        n_repeats=6,
        norm_type="gLN",
        ff_activation="relu",
        bidirectional=True,
        dropout=0,
    ):
        super(DPTEncoder, self).__init__()
        self.in_chan = in_chan
        self.n_heads = n_heads
        self.ff_hid = ff_hid
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.norm_type = norm_type
        self.ff_activation = ff_activation
        self.bidirectional = bidirectional
        self.dropout = dropout

        if ff_activation == 'relu':
            self.ff_activation = nn.ReLU
        else:
            raise NotImplementedError

        # Get normalization
        if norm_type == 'gLN':
            self.norm_type = GlobLN
        else:
            raise NotImplementedError

        # Multihead attention specifics
        self.mha_in_dim = ceil(self.in_chan / self.n_heads) * self.n_heads
        if self.in_chan % self.n_heads != 0:
            warnings.warn(
                f"DPTEncoder input dim ({self.in_chan}) is not a multiple of the number of "
                f"heads ({self.n_heads}). Adding extra linear layer at input to accomodate "
                f"(size [{self.in_chan} x {self.mha_in_dim}])"
            )
            self.input_layer = nn.Linear(self.in_chan, self.mha_in_dim)
        else:
            self.input_layer = None

        self.in_norm = self.norm_type(self.mha_in_dim)
        self.ola = DualPathProcessing(self.chunk_size, self.hop_size)

        # Succession of dual-path transformer blocks.
        self.layers = nn.ModuleList([])
        for x in range(self.n_repeats):
            self.layers.append(
                nn.ModuleList(
                    [
                        ImprovedTransformedLayer(
                            self.mha_in_dim,
                            self.n_heads,
                            self.ff_hid,
                            self.dropout,
                            self.ff_activation,
                            True,
                            self.norm_type,
                        ),
                        ImprovedTransformedLayer(
                            self.mha_in_dim,
                            self.n_heads,
                            self.ff_hid,
                            self.dropout,
                            self.ff_activation,
                            self.bidirectional,
                            self.norm_type,
                        ),
                    ]
                )
            )

        # Final layers
        self.first_out = nn.Sequential(
                            nn.PReLU(),
                            nn.Conv2d(self.mha_in_dim, self.in_chan, 1)
                            )

        ## Gating and masking in 2D space (after fold)
        #self.net_out = nn.Sequential(nn.Conv1d(self.in_chan, self.in_chan, 1), nn.Tanh())
        #self.net_gate = nn.Sequential(nn.Conv1d(self.in_chan, self.in_chan, 1), nn.Sigmoid())

    def forward(self, mixture_w):
        r"""Forward.

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$

        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        if self.input_layer is not None:
            mixture_w = self.input_layer(mixture_w.transpose(1, 2)).transpose(1, 2)

        mixture_w = self.in_norm(mixture_w) # (batch, mha_in_dim, n_frames)
        n_orig_frames = mixture_w.shape[-1]

        mixture_w = self.ola.unfold(mixture_w)
        batch, n_filters, self.chunk_size, n_chunks = mixture_w.size()

        for layer_idx in range(len(self.layers)):
            intra, inter = self.layers[layer_idx]
            mixture_w = self.ola.intra_process(mixture_w, intra)
            mixture_w = self.ola.inter_process(mixture_w, inter)

        output = self.first_out(mixture_w) # (batch, in_chan, n_frames)
        output = output.reshape(batch, self.in_chan, self.chunk_size, n_chunks)
        output = self.ola.fold(output, output_size=n_orig_frames)

        #output = self.net_out(output) * self.net_gate(output)

        output = output.reshape(batch, self.in_chan, -1)
        return output


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
