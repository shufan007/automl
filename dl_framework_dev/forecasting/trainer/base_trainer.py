
import os
import shutil
import json
import torch
from models import FEDformer, Autoformer, Transformer, Informer
from common import get_logger, copy_dir, get_suffix_file

logger = get_logger()


class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
        }

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        devices = None
        self.args.device_ids = []
        if torch.cuda.is_available():
            devices = os.getenv("CUDA_VISIBLE_DEVICES")
            logger.info(f"CUDA_VISIBLE_DEVICES: {devices}")
        if devices is not None:
            device_ids = devices.replace(' ', '').split(',')
            # device_ids = [int(i) for i in device_ids]
            self.args.device_ids = list(range(0, len(device_ids)))
            # gpu = device_ids[0]
            gpu = 0
            device = torch.device('cuda:{}'.format(gpu))
            logger.info('Use GPU: cuda:{}'.format(gpu))
        else:
            device = torch.device('cpu')
            logger.info('Use CPU')
        self.args.device = device
        return device

    def eval(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass

    def save_model(self, checkpoint_dir, model_path):
        """
        save model to torchscript
        :param checkpoint_dir:
        :param model_path:
        :return:
        """
        pth_model_path = os.path.join(model_path, 'pth_model')
        if not os.path.exists(pth_model_path):
            os.makedirs(pth_model_path)

        """ save args """
        json_save_path = os.path.join(pth_model_path, "args.json")
        with open(json_save_path, 'w', encoding='utf-8') as file:
            self.args.device = str(self.args.device)
            json.dump(self.args, file, ensure_ascii=False)

        des = None
        if checkpoint_dir is not None:
            model_name = get_suffix_file(checkpoint_dir, '.pth')
            src = os.path.join(checkpoint_dir, model_name)
            des = os.path.join(pth_model_path, "model.pth")
            logger.info(f'copy model from {src} to {des}')
            shutil.copy(src, des)

        logger.info(f'copy model definitions to {pth_model_path}')
        for src in self.args.model_definition_path:
            dst = os.path.join(pth_model_path, os.path.basename(src))
            copy_dir(src, dst, ignore_pattern="__pycache__")

        logger.info(" ###### Save model to TorchScript #####")
        model = self.model_dict[self.args.model].Model(self.args).float()

        if des is not None:
            logger.info(f'loading model from {des}')
            model.load_state_dict(torch.load(des))
            model.eval()
            torchscript_file = os.path.join(model_path, 'model.torchscript')
            logger.info("Save TorchScript model to: {}".format(model_path))
            traced = torch.jit.script(model)
            traced.save(torchscript_file)
