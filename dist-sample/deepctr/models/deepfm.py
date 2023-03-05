import torch
import torch.nn as nn
from ..layers.layers import FM, Linear, DNN


class DeepFM(nn.Module):
    def __init__(self,
                 feature_cols,
                 sparse_cols,
                 num_embeddings,
                 init_std=0.0001,
                 hidden_units=[256, 128],
                 linear_embedding_dim=1,
                 fm_embedding_dim=10):
        super(DeepFM, self).__init__()

        self.feature_cols = feature_cols
        self.sparse_cols = sparse_cols
        self.init_std = init_std

        self.linear_emb = nn.Embedding(num_embeddings, linear_embedding_dim)
        self.linear_layer = Linear(self.feature_cols, self.sparse_cols, self.linear_emb, init_std)

        self.fm_emb = nn.Embedding(num_embeddings, fm_embedding_dim)
        self.fm_layer = FM()

        dense_feature_dim = len(feature_cols) - len(sparse_cols)
        sparse_feature_dim = len(sparse_cols) * fm_embedding_dim
        self.dnn_layer = nn.Sequential(
            DNN(dense_feature_dim + sparse_feature_dim, (hidden_units[0], hidden_units[1]), init_std=init_std),
            nn.Linear(hidden_units[1], 1, bias=False))

        self.bias = nn.Parameter(torch.zeros((1,)))
        self.apply(self.weight_init)

    def weight_init(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(std=self.init_std)

    def forward(self, x):
        sparse_feature_list = []
        for col in self.sparse_cols:
            index = self.feature_cols.index(col)
            input = x[:, index:index + 1].long()
            sparse_feature_list.append(self.fm_emb(input))

        # fm
        fm_input = torch.cat(sparse_feature_list, dim=1)
        logit = self.fm_layer(fm_input)

        # linear
        logit += self.linear_layer(x)

        # dnn
        dense_feature_list = []
        for index, col in enumerate(self.feature_cols):
            if col not in self.sparse_cols:
                input = x[:, index:index + 1].float()
                dense_feature_list.append(input)

        sparse_dnn_input = torch.flatten(torch.cat(sparse_feature_list, dim=-1), start_dim=1)
        dense_features = torch.flatten(torch.cat(dense_feature_list, dim=-1), start_dim=1)
        dnn_input = torch.cat([sparse_dnn_input, dense_features], dim=-1)
        dnn_logit = self.dnn_layer(dnn_input)
        logit += dnn_logit

        output = logit + self.bias
        y_pred = torch.sigmoid(output)

        return y_pred
