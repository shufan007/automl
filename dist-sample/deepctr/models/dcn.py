import torch
import torch.nn as nn
from .basemodel import BaseModel, Linear
from ..layers.interaction import CrossNet
from ..layers.core import DNN


class DCN(BaseModel):
    def __init__(self, feature_list, embedding_dimension_dict, cross_num=3,
                 cross_parameterization='vector', init_std=0.0001, device='cpu'):
        super(DCN, self).__init__(feature_list, embedding_dimension_dict, init_std=init_std, device=device)
        self.feature_list = feature_list
        self.device = device
        self.cross_num = cross_num
        self.linear_model = Linear(feature_list, embedding_dimension_dict, init_std, device=device)
        self.crossnet = CrossNet(in_features=273,
                                 layer_num=cross_num, parameterization=cross_parameterization, device=device)
        self.dnn = DNN(273, (128, 128), init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(401, 1, bias=False).to(device)
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
            if 'I' in list(feature):
                dense_value_list.append(input)
        logit = self.linear_model(X)

        sparse_dnn_input = torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
        dnn_input = torch.cat([sparse_dnn_input, dense_dnn_input], dim=-1)
        dnn_output = self.dnn(dnn_input)
        cross_out = self.crossnet(dnn_input)
        stack_out = torch.cat((cross_out, dnn_output), dim=-1)
        logit += self.dnn_linear(stack_out)

        output = logit + self.bias
        y_pred = torch.sigmoid(output)
        return y_pred
