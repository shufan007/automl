import json
from types import SimpleNamespace
import torch
from models import FEDformer, Autoformer, Transformer, Informer

model_dict = {
    'FEDformer': FEDformer,
    'Autoformer': Autoformer,
    'Transformer': Transformer,
    'Informer': Informer,
}

json_path = "./args.json"
with open(json_path, 'r', encoding='utf-8') as file:
    args = json.load(file)

args = SimpleNamespace(**args)
model = model_dict[args.model].Model(args)

model_path = './model.pth'
model.load_state_dict(torch.load(model_path))
