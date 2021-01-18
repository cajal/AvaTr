import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .light_wav2vec2 import Wav2Vec2Model, DeConvModel


class AvaTr(nn.Module):
    def __init__(
        self,
        # Avatars
        n_spk,
        # Wav2Vec2 encoder
        w2v_ckpt='/mnt/scratch07/hushell/UploadAI/ckpts/wav2vec_small.pt',
        freeze_w2v=False,
        # DeTr decoder
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False):

        super().__init__()

        # Wav2Vec2 encoder
        w2v_dict = torch.load(w2v_ckpt)
        w2v_dict['args'].encoder_embed_dim = 512
        w2v_dict['args'].encoder_attention_heads = 8
        w2v_dict['args'].encoder_ffn_embed_dim = 2048
        w2v_dict['args'].encoder_layers = 6
        w2v_dict['args'].encoder_normalize_before = False
        self.wav2vec = Wav2Vec2Model.build_model(w2v_dict['args'])
        #self.wav2vec.load_state_dict(w2v_dict['model'], strict=False)
        #if freeze_w2v:
        #    for param in self.wav2vec.parameters():
        #        param.requires_grad = False

        embed_dim = w2v_dict['args'].encoder_embed_dim
        d_model = w2v_dict['args'].encoder_embed_dim

        # Avatars
        self.avatar = nn.Embedding(n_spk, embed_dim)

        # Decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        #decoder_norm = nn.LayerNorm(d_model)
        decoder_norm = None
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self.mask_act = nn.Sigmoid()

        # DeConv
        #self.post_extract_unproj = nn.Linear(d_model, 512)
        #self.post_extract_norm = nn.LayerNorm(512)
        self.deconv = DeConvModel(
            conv_layers=[(512, 3, 2)] * 6 + [(1, 10, 5)],
            dropout=0.0,
        )

    def forward(self, inputs, mask=None):
        mix, spk_id = inputs

        # conv + wav2vec encoding
        results = self.wav2vec(mix, padding_mask=mask)
        mix_feat = results['x'].transpose(0, 1) # T x B x C
        T, B, C = mix_feat.shape
        pos_embed = results['pos_embed'].transpose(0, 1) # T x B x C

        # avatar embedding
        query_embed = self.avatar(spk_id) # B x 512
        query_embed = query_embed.unsqueeze(0).repeat(T, 1, 1) # T x B x C

        # mask generation
        tgt = torch.ones_like(query_embed)
        mix_mask = self.decoder(tgt, mix_feat, memory_key_padding_mask=mask,
                            pos=pos_embed, query_pos=query_embed) # T x B x C
        mix_mask = self.mask_act(mix_mask)

        # masking
        hs = mix_feat * mix_mask

        # deconv
        #hs = self.post_extract_unproj(hs)
        #hs = self.post_extract_norm(hs)
        hs = hs.permute(1, 2, 0)
        signal = self.deconv(hs)

        return signal.squeeze(1)


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
