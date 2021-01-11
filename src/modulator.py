import torch
import torch.nn as nn


class SigmoidDotAttention(nn.Module):
    """ Applies attention on encoder output using avater as the query

    Args:
        dim (int): the number of expected features in the output
        embedding_dim (int): the size of avatar embedding
    Inputs:
        sequence: (batch, seq_len, dimensions): tensor containing the output features from the decoder.
        avater_emb: (batch, dimensions): tensor containing features of the encoded input sequence.
    Outputs:
        attn: (batch, score_len, seq_len): tensor containing attention weights.
    """
    def __init__(self, dim=128, embedding_dim=512):
        super(SigmoidDotAttention, self).__init__()
        self.embedding_linear = nn.Linear(embedding_dim, dim, bias=True)
        self.sequence_linear = nn.Linear(dim, dim, bias=True)

    def forward(self, sequence, ava_emb):

        b, c, s = sequence.shape

        sequence = sequence.transpose(1, 2)
        sequence = self.sequence_linear(sequence) # b, s, dim

        q = self.embedding_linear(ava_emb).unsqueeze(-1)

        # (b, s, dim) * (b, dim, 1) -> (b, s, 1)
        attn = torch.bmm(sequence, q)
        attn = torch.sigmoid(attn)

        output = sequence * attn

        return output.transpose(1, 2), attn


class Modulator(nn.Module):
    """ Using avatar to modulate mixture feature

    Args:
        n_spk (int): number of speakers
        feat_dim (int): size of mixture feature
        embed_dim (int): size of avatar embedding
    Inputs:
        mixture_feature (B, C, T): feature of the mixed audio
        spk_id (B, embed_dim): speaker id
    Outputs:
        mixture_feature (B, n_src, C, T): modulated feature
    """
    def __init__(self, n_spk, feat_dim, embed_dim):
        super(Modulator, self).__init__()

        self.avatar = nn.Embedding(n_spk, embed_dim)
        self.att = SigmoidDotAttention(feat_dim, embed_dim)

    def forward(self, mixture_feature, spk_id):
        if not isinstance(spk_id, list):
            spk_id = [spk_id]

        mixture_feature_t = []

        for component in spk_id:
            dvec = self.avatar(component) # B x emb_dim
            mix_f, _ = self.att(mixture_feature, dvec) # B x feat_dim x T
            mixture_feature_t.append(mix_f)

        mixture_feature_t = torch.stack(mixture_feature_t, dim=1) # B x n_src x feat_dim x T

        return mixture_feature_t
