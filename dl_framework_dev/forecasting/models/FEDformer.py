import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, \
    series_decomp_multi

from typing import Optional

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.complex_split = True if configs.use_amp else False
        self.d_out = configs.d_out
        # print(self.complex_split)
        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        if configs.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  n_heads=configs.n_heads,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=configs.modes,
                                                  ich=configs.d_model,
                                                  base=configs.base,
                                                  activation=configs.cross_activation)
        else:
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            n_heads=configs.n_heads,
                                            seq_len=self.seq_len,
                                            modes=configs.modes,
                                            mode_select_method=configs.mode_select,
                                            complex_split=self.complex_split)
            decoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            n_heads=configs.n_heads,
                                            seq_len=self.seq_len // 2 + self.pred_len,
                                            modes=configs.modes,
                                            mode_select_method=configs.mode_select,
                                            complex_split=self.complex_split)
            decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
                                                      out_channels=configs.d_model,
                                                      n_heads=configs.n_heads,
                                                      seq_len_q=self.seq_len // 2 + self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=configs.modes,
                                                      mode_select_method=configs.mode_select,
                                                      complex_split=self.complex_split)
        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len // 2))
        dec_modes = int(min(configs.modes, (configs.seq_len // 2 + configs.pred_len) // 2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ffn,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.enc_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_out,
                    configs.d_ffn,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.dec_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.d_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc: Optional[torch.Tensor] = None,
                x_mark_dec: Optional[torch.Tensor] = None,
                enc_self_mask: Optional[torch.Tensor] = None,
                dec_self_mask: Optional[torch.Tensor] = None,
                dec_enc_mask: Optional[torch.Tensor] = None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        # zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # dec
        if (x_mark_enc is not None) and (x_mark_dec is not None):
            x_mark_dec = torch.cat([x_mark_enc[:, -self.label_len:, :], x_mark_dec], dim=1).float().to(x_enc.device)

        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend=trend_init,
                                                 x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        # final
        dec_out = seasonal_part
        if trend_part is not None:
            dec_out = dec_out + trend_part[..., -self.d_out:]

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :], None  # [B, L, D]
