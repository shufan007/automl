
import os
from typing import Optional
import numpy as np
from torch.utils.data import Dataset
from flaml import AutoML as FlamlAutoML

from autotabular.common import get_logger
from autotabular.common.base import BaseTrainer
from autotabular.common.base.utils import (
    build_train_settings,
    get_search_space,
    save_feature_importance,
    save_learning_curve
)
from autotabular.common.base.constants import MODEL_PARAMS_FILE

import warnings
warnings.filterwarnings('ignore')

logger = get_logger()


class AutoClassifier(BaseTrainer):
    def __init__(self,
                 args,
                 train_dataset: Optional[Dataset] = None,
                 val_dataset: Optional[Dataset] = None):
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        assert self.args.output_dir is not None, "'output_dir' must given."

        self.train_settings = build_train_settings(self.args)
        if self.args.hp_domain is not None:
            custom_hp = get_search_space(self.args.hp_domain)
            self.train_settings["custom_hp"] = custom_hp

        self.larger_better_loss_list = ['accuracy', 'roc_auc', 'roc_auc_ovr',
                                        'roc_auc_ovo', 'f1', 'ap', 'micro_f1', 'macro_f1']
        if self.args.metric_constraint is not None:
            sign = ">=" if self.args.metric in self.larger_better_loss_list else "<="
            self.train_settings["metric_constraints"] = [(self.args.metric, sign, self.args.metric_constraint)]

        self.auto_trainer = FlamlAutoML()

    def train(self):
        logger.info('Train model by auto trainer')
        (X_train, y_train) = self.train_dataset.to_ndarray()

        if self.val_dataset is not None:
            (X_val, y_val) = self.val_dataset.to_ndarray()
            self.auto_trainer.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **self.train_settings)
        else:
            self.auto_trainer.fit(X_train=X_train, y_train=y_train, **self.train_settings)

        logger.info('Retrieve best config and best learner')
        logger.info('Best ML leaner: {}'.format(self.auto_trainer.best_estimator))
        logger.info('Best hyper parameter config:{}'.format(self.auto_trainer.best_config))
        if self.auto_trainer.best_config_train_time:
            logger.info('Training duration of best run: {0:.4g} s'.format(self.auto_trainer.best_config_train_time))

        best_eval = self.auto_trainer.best_loss
        if self.train_settings["metric"] in self.larger_better_loss_list:
            best_eval = 1 - best_eval
        logger.info('Best {0} : {1:.4g}'.format(self.train_settings["metric"], best_eval))

    def eval(self):
        (X_val, y_val) = self.val_dataset.to_ndarray()

        logger.info("Evaluation on validation dataset")
        y_pred = self.auto_trainer.predict(X_val)
        y_pred_proba = self.auto_trainer.predict_proba(X_val)[:, 1]
        from flaml.ml import sklearn_metric_loss_score
        logger.info('Accuracy = {}'.format(1 - sklearn_metric_loss_score('accuracy', y_pred, y_val)))

        n_class = len(np.unique(y_val))
        if n_class == 2:
            logger.info('roc_auc = {}'.format(1 - sklearn_metric_loss_score('roc_auc', y_pred_proba, y_val)))
            logger.info('f1 = {}'.format(1 - sklearn_metric_loss_score('f1', y_pred, y_val)))
        else:
            logger.info('micro_f1 = {}'.format(1 - sklearn_metric_loss_score('micro_f1', y_pred, y_val)))
            logger.info('macro_f1 = {}'.format(1 - sklearn_metric_loss_score('macro_f1', y_pred, y_val)))

    def save_model(self, output_dir: Optional[str] = None):
        if output_dir is None:
            output_dir = self.args.output_dir

        params_file = os.path.join(output_dir, MODEL_PARAMS_FILE)
        logger.info("Save model params to file: {}".format(params_file))
        self.auto_trainer.save_best_config(params_file)

        model_file = os.path.join(output_dir, self.auto_trainer.best_estimator + '.model')
        logger.info("Save model to: {}".format(model_file))
        if self.auto_trainer.best_estimator == 'lgbm':
            self.auto_trainer.model.estimator.booster_.save_model(model_file)
        elif self.auto_trainer.best_estimator == 'xgboost':
            self.auto_trainer.model.estimator.save_model(model_file)

        save_feature_importance(self.args, self.auto_trainer.model.estimator.feature_importances_)
        save_learning_curve(self.args, self.train_settings, self.larger_better_loss_list)

