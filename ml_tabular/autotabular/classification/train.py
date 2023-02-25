
import json
import argparse

from autotabular.classification.classifiers import AutoClassifier
from autotabular.common.data import TabularDataset
from autotabular.common import MultiNodeRunner, get_logger, dotdict, get_luban_node_resources

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AutoML Classifier')
    # env args
    parser.add_argument('--hostfile', type=str, help='the host file')

    # data args
    parser.add_argument('--train_input', type=str, help='train input')
    parser.add_argument('--val_input', type=str, help='[optional] validation input')
    parser.add_argument('--output_dir', type=str, help='output directory for model to save')
    parser.add_argument('--data_format', type=str, default='table', choices=['table', 'parquet', 'orc', 'csv'],
                        help='train data format, "table": hive table name, others: file(s) path of tabular data.')
    parser.add_argument('--feature_cols', type=str, help='[optional] features column list')
    parser.add_argument('--label_cols', type=str, default='label', help='data label, default:"label"')

    # trainer args
    parser.add_argument('--time_budget', type=int, default=-1,
                        help='constrains the wall-clock time (seconds) used by the AutoML process. Use -1 if no time limit.')
    parser.add_argument('--seed', type=int, default=0, help='seed, default:0')
    parser.add_argument('--metric', type=str, default='log_loss',
                        choices=['accuracy', 'log_loss', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'f1', 'micro_f1', 'macro_f1'],
                        help='optimization metric')
    parser.add_argument('--estimator_list', type=str, default='["lgbm"]',
                        help=""" estimator list:
                             'lgbm': LGBMEstimator for task "classification", "regression". 
                             'xgboost': XGBoostSkLearnEstimator for task "classification", "regression".
                             """)
    parser.add_argument('--hp_domain', type=str,
                        help=""" hyperparameters domain, params dict like:
                        '{"n_estimators":[4,1024],"max_leaves":[4,1024],"learning_rate":[0.001,1.0],
                        "colsample_bytree":[0.01,1.0],"reg_alpha":[0.001,1000],"reg_lambda":[0.001,1000]}'
                             """)
    parser.add_argument('--fit_kwargs', type=str,
                        help="""[optional] other fit kwargs: params dict for the given estimator.
                        eg:'{"n_jobs":2,"n_concurrent_trials":2,"min_sample_size":10000,"retrain_full":0,"log_type":"all",
                        "max_iter":100,"eval_method":"holdout","split_ratio":0.1}'
                        """)
    parser.add_argument('--node_resources', type=str,
                        help="""node_resources: params dict for the node resources,like:
                        '{"cpu":4,"memory":32,"object_store_memory":10}'
                        cpu: INTEGER, cpu limit of one node
                        memory: memory limit of one node, unit:GBi
                        object_store_memory: ray object store memory, unit:GBi
                        """)

    args = parser.parse_args()
    args = dotdict(vars(args))
    args = prepare_args(args)

    return args


def prepare_args(args):
    if args.train_input is None:
        raise FileExistsError("Error! Train dataset must be given.")

    if args.feature_cols:
        args.feature_cols = eval(args.feature_cols)
    if args.label_cols:
        args.label_cols = eval(args.label_cols)

    if args.estimator_list:
        args.estimator_list = eval(args.estimator_list)
    if args.fit_kwargs:
        args.fit_kwargs = eval(args.fit_kwargs)
    else:
        args.fit_kwargs = dict()

    if args.hp_domain:
        args.hp_domain = json.loads(args.hp_domain)
        for hp, domain in args.hp_domain.items():
            assert domain[1] >= domain[0], \
                f"Error, search space args, {hp}:{domain}, the lower should no bigger than the upper!"

    args.fit_kwargs = dotdict(args.fit_kwargs)
    if args.fit_kwargs.n_jobs is None:
        args.fit_kwargs["n_jobs"] = -1
    if args.fit_kwargs.n_concurrent_trials is None:
        args.fit_kwargs["n_concurrent_trials"] = 1
        if args.hostfile is not None:
            runner = MultiNodeRunner(hostfile=args.hostfile)
            args.fit_kwargs["n_concurrent_trials"] = len(runner.nodes)

    n_cpus, memory = get_luban_node_resources()
    logger.info('n_cpus on head node: {}'.format(n_cpus))
    if n_cpus is not None:
        if args.fit_kwargs["n_jobs"] == -1 or args.fit_kwargs["n_jobs"] > n_cpus:
            args.fit_kwargs["n_jobs"] = n_cpus

    if args.node_resources:
        args.node_resources = eval(args.node_resources)
    else:
        args.node_resources = dict()
    args.node_resources = dotdict(args.node_resources)
    if n_cpus is not None:
        args.node_resources.cpu = n_cpus
    if memory is not None:
        args.node_resources.memory = memory

    if args.search_space:
        args.search_space = json.loads(args.search_space)

    args.task = 'classification'

    return args


def train_classifier(args):
    # startup ray cluster
    runner = MultiNodeRunner(hostfile=args.hostfile, node_resources=args.node_resources)
    runner.start()

    # dataset
    train_dataset = TabularDataset(path=args.train_input,
                                   format=args.data_format,
                                   feature_cols=args.feature_cols,
                                   label_cols=args.label_cols)
    val_dataset = None
    if args.val_input is not None:
        val_dataset = TabularDataset(path=args.val_input,
                                     format=args.data_format,
                                     feature_cols=args.feature_cols,
                                     label_cols=args.label_cols)

    # train process
    classifier = AutoClassifier(args, train_dataset, val_dataset)
    classifier.train()

    # evaluation
    if val_dataset:
        classifier.eval()

    # save model
    classifier.save_model()

    # stop ray cluster
    runner.stop()


def main():
    # parse args
    args = parse_args()
    train_classifier(args)


if __name__ == '__main__':
    main()

