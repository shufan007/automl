import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding
from typing import Optional


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ffn,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.enc_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.enc_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ffn,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.dec_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc: Optional[torch.Tensor]=None,
                x_mark_dec: Optional[torch.Tensor]=None,
                enc_self_mask: Optional[torch.Tensor]=None,
                dec_self_mask: Optional[torch.Tensor]=None,
                dec_enc_mask: Optional[torch.Tensor]=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # decoder input
        x_dec = F.pad(x_enc[:, -self.label_len:, :], (0, 0, 0, self.pred_len))

        if (x_mark_enc is not None) and (x_mark_dec is not None):
            x_mark_dec = torch.cat([x_mark_enc[:, -self.label_len:, :], x_mark_dec], dim=1).float().to(x_enc.device)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :], None  # [B, L, D]
