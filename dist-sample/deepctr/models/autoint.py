import torch
import torch.nn as nn
from .basemodel import BaseModel, Linear
from ..layers.interaction import InteractingLayer
from ..layers.core import DNN


class AutoInt(BaseModel):
    def __init__(self, feature_list, embedding_dimension_dict, att_layer_num=3,
                 att_head_num=2, att_res=True, init_std=0.0001, device='cpu'):
        super(AutoInt, self).__init__(feature_list, embedding_dimension_dict, init_std=init_std, device=device)
        self.linear_model = Linear(feature_list, embedding_dimension_dict, init_std, device=device)
        self.int_layers = nn.ModuleList(
            [InteractingLayer(10, att_head_num, att_res, device=device) for _ in range(att_layer_num)])

        self.dnn = DNN(273, (256, 128), init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(388, 1, bias=False).to(device)
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.to(device)

    def forward(self, X):
        sparse_embedding_list = []
        dense_value_list = []
        for key, fun in self.embedding_dict.items():
            index = self.feature_list.index(key)
            input = X[:, index:index + 1]
            max_tensor = torch.ones_like(input) * fun.num_embeddings - 1
            input = torch.where(input > fun.num_embeddings - 1, max_tensor, input)
            sparse_embedding_list.append(fun(input.long()))

        for index, feature in enumerate(self.feature_list):
            input = X[:, index:index + 1]
            if 'c' in feature:
                dense_value_list.append(input)
        logit = self.linear_model(X)

        att_input = torch.cat(sparse_embedding_list, dim=1)
        for index, layer in enumerate(self.int_layers):
            att_input = layer(att_input)
        att_output = torch.flatten(att_input, start_dim=1)

        sparse_dnn_input = torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
        dnn_input = torch.cat([sparse_dnn_input, dense_dnn_input], dim=-1)
        dnn_output = self.dnn(dnn_input)
        stack_out = torch.cat([att_output, dnn_output], dim=-1)
        logit += self.dnn_linear(stack_out)

        output = logit + self.bias
        y_pred = torch.sigmoid(output)
        return y_pred
