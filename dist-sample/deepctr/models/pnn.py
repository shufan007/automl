import torch
import torch.nn as nn
from .basemodel import BaseModel
from ..layers.interaction import InnerProductLayer, OutterProductLayer
from ..layers.core import DNN


class PNN(BaseModel):
    def __init__(self, feature_list, embedding_dimension_dict, use_inner=True, use_outter=False, kernel_type='mat',
                 init_std=0.0001, device='cpu'):
        super(PNN, self).__init__(feature_list, embedding_dimension_dict, init_std=init_std, device=device)
        if kernel_type not in ['mat', 'vec', 'num']:
            raise ValueError("kernel_type must be mat,vec or num")
        self.kernel_type = kernel_type
        self.use_inner = use_inner
        self.use_outter = use_outter

        product_out_dim = 0
        num_inputs = len(embedding_dimension_dict)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        if self.use_inner:
            product_out_dim += num_pairs
            self.innerproduct = InnerProductLayer(device=device)

        if self.use_outter:
            product_out_dim += num_pairs
            self.outterproduct = OutterProductLayer(
                num_inputs, 10, kernel_type=kernel_type, device=device)

        self.dnn = DNN(273 + product_out_dim, (256, 128), init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(128, 1, bias=False).to(device)
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
            if 'I' in feature:
                dense_value_list.append(input)
        linear_signal = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1)

        if self.use_inner:
            inner_product = torch.flatten(
                self.innerproduct(sparse_embedding_list), start_dim=1)

        if self.use_outter:
            outer_product = self.outterproduct(sparse_embedding_list)

        if self.use_outter and self.use_inner:
            product_layer = torch.cat(
                [linear_signal, inner_product, outer_product], dim=1)
        elif self.use_outter:
            product_layer = torch.cat([linear_signal, outer_product], dim=1)
        elif self.use_inner:
            product_layer = torch.cat([linear_signal, inner_product], dim=1)
        else:
            product_layer = linear_signal

        sparse_dnn_input = torch.flatten(torch.cat([product_layer], dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
        dnn_input = torch.cat([sparse_dnn_input, dense_dnn_input], dim=-1)
        dnn_output = self.dnn(dnn_input)
        logit = self.dnn_linear(dnn_output)

        output = logit + self.bias
        y_pred = torch.sigmoid(output)
        return y_pred
