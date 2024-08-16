""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""
import os
import warnings
import logging
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from argparse import ArgumentParser, Namespace
import sys
sys.path.append(sys.path[0]+'/../..')
print(sys.path)
from fine_tune.model import MInterface  # noqa: E402
from fine_tune.data import DInterface  # noqa: E402
from fine_tune.load_model import load_model_path_by_args  # noqa: E402
# solve "Some weights of the model checkpoint at bert-base-uncased were not used"
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
# ignore all warnings
warnings.filterwarnings('ignore')


class CustomModelCheckpoint(Callback):
    def __init__(self, filename: str = "checkpoints/best_model.ckpt"):
        super().__init__()
        self.filename = filename
        self.highest_f1 = 0.0
        self.best_model_path = None
        self.validation_called = False

    def on_validation_end(self, trainer, pl_module):
        # Used to skip the initial Validation sanity check
        if not self.validation_called:
            self.validation_called = True
            return
        f1 = trainer.callback_metrics.get('epoch_all_f1', 0.0)
        recall = trainer.callback_metrics.get('epoch_all_recall', 0.0)
        save_dir = trainer.log_dir
        if recall > 80 and f1 > self.highest_f1:
            self.highest_f1 = f1
            self.best_model_path = os.path.join(save_dir, self.filename)
            trainer.save_checkpoint(self.best_model_path)
            print("\nA new model is saved\n")
            print("ok")


def build_args():
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--label_set', action='append', default=None)
    parser.add_argument('--mode', type=str, default='train+test',
                        choices=['train', 'test', 'train+test', 'predict', 'outer_predict'])
    parser.add_argument('--input_method', type=int, default=3)
    parser.add_argument('--output_method', type=int, default=1)

    # Restart training Control
    parser.add_argument('--load_best', action='store_true', default=True)
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--load_ver', type=str, default=None)
    parser.add_argument('--load_v_num', type=int, default=0)

    # Training Dataset Info
    parser.add_argument('--dataset', type=str, default='sentence_data')
    parser.add_argument('--train_data_dir', type=str, default=None)
    parser.add_argument('--val_data_dir', type=str, default=None)
    parser.add_argument('--test_data_dir', type=str, default=None)
    parser.add_argument('--predict_data_dir', type=str, default=None)
    parser.add_argument('--kfold', type=int, default=0)
    parser.add_argument('--fold_num', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--vec_max_len', type=int, default=512)

    # Training Model Info
    parser.add_argument('--model_name', type=str, default='bert_classifier')
    parser.add_argument('--is_pretrain_model', type=bool, default=False)
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='dmis-lab_biobert-base-cased-v1.2')
    parser.add_argument('--pretrain_model_parameter', action='append', default=None)
    # , type=list, default=['pretrained_model_name_or_path', 'num_labels']

    # optimizer
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--encoder_lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--eps', type=float, default=1e-6)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=['step', 'cosine', 'warmup'])
    parser.add_argument('--lr_decay_steps', type=int, default=20)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_min_lr', type=float, default=1e-5)

    # loss and validation, val_list:['acc', 'f1', 'f_beta', 'eval_entity', 'eval_subtag']
    parser.add_argument('--loss_cal_type', type=str, default='loss_f', choices=['model_out', 'loss_f', 'model_f'])
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--val_list', action='append', default=None)

    # frozen
    parser.add_argument('--nr_frozen_epochs', default=1, type=int)

    # Other
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--aug_prob', type=float, default=0.5)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--logger_type', type=str, default='TensorBoard', choices=['TensorBoard', 'Wandb'])
    parser.add_argument('--log_dir', type=str, default='lightning_logs')
    parser.add_argument('--logger_name', type=str, default='lightning_classification')
    parser.add_argument('--logger_project', type=str, default='hedges')

    # 自动添加所有Trainer会用到的命令行参数
    # Reset Some Default Trainer Arguments' Default Values
    parser = Trainer.add_argparse_args(parser)
    # .add_argument_group(title="pl.Trainer args")
    parser.set_defaults(max_epochs=100)
    parser.set_defaults(gpus=1)
    parser.set_defaults(accumulate_grad_batches=2)
    # parser.set_defaults(fast_dev_run=False)
    # parser.set_defaults(progress_bar_refresh_rate=0)
    # parser.set_defaults(num_sanity_val_steps=0)

    args = parser.parse_args()
    return args


def main(args):
    pl.seed_everything(args.seed)
    data_module = DInterface(**vars(args))
    model = MInterface(**vars(args))
    if args.logger_type == 'TensorBoard':
        args.logger = TensorBoardLogger(save_dir=args.log_dir, name="")
    elif args.logger_type == 'Wandb':
        args.logger = WandbLogger(name=args.logger_name,
                                  # group=args.logger_group,
                                  project=args.logger_project,
                                  save_dir=args.log_dir,
                                  offline=True,)
    else:
        ValueError('logger_type error')

    if args.mode == 'train':
        callbacks = load_callbacks(args)
        args.callbacks = callbacks
        trainer = Trainer.from_argparse_args(args)  # Trainer直接从args中加载，如果no在命令行中修改，就使用默认值
        result = trainer.fit(model, data_module)
    elif args.mode == 'test':
        checkpoint_path = load_model_path_by_args(args)
        model = model.load_from_checkpoint(checkpoint_path=checkpoint_path)
        model.eval()
        model.freeze()
        trainer = Trainer.from_argparse_args(args)  # Trainer直接从args中加载，如果no在命令行中修改，就使用默认值
        result = trainer.test(model, datamodule=data_module)
    elif args.mode == 'train+test':
        callbacks = load_callbacks(args)
        args.callbacks = callbacks
        trainer = Trainer.from_argparse_args(args)  # Trainer直接从args中加载，如果no在命令行中修改，就使用默认值
        trainer.fit(model, data_module)
        model.eval()
        model.freeze()
        result = trainer.test(model, datamodule=data_module)
    elif args.mode == 'predict':
        checkpoint_path = load_model_path_by_args(args)
        model = model.load_from_checkpoint(checkpoint_path=checkpoint_path)  #
        model.eval()
        model.freeze()
        trainer = Trainer.from_argparse_args(args)  # Trainer直接从args中加载，如果no在命令行中修改，就使用默认值
        result = trainer.predict(model, datamodule=data_module)
        print(result)
    elif args.mode == 'outer_predict':
        checkpoint_path = load_model_path_by_args(args)
        model = model.load_from_checkpoint(checkpoint_path=checkpoint_path)  #
        model.eval()
        model.freeze()
        sample = [{'sentence': 'other metrics that are already implemented vs to-do'},
                  {'sentence': 'To quickly get started with local development'},
                  {'sentence': "EGFR TKI pre-treated patients whose T790M mutation status cannot be determined."}
                  ]
        predict_dataloader = data_module.predict_dataloader(predict_data=sample)
        trainer = Trainer.from_argparse_args(args)
        result = trainer.predict(model, dataloaders=predict_dataloader)
        print(result)
    return result


def load_callbacks(args):
    callbacks = [
        plc.EarlyStopping(
            monitor='epoch_all_f1',
            mode='max',
            patience=5,
            verbose=True,
            min_delta=0.001
        ),
        plc.ModelCheckpoint(
            monitor='epoch_all_recall',
            filename='best-{epoch}-{epoch_all_f1:.3f}',  # :02d
            save_top_k=0,
            mode='max',
            save_last=False
        ),
        CustomModelCheckpoint()
    ]
    if args.lr_scheduler is not None:
        callbacks.append(plc.LearningRateMonitor(logging_interval='epoch'))
    return callbacks


def namespace_add_item_from_dic(args, config):
    args_dic = vars(args)
    args_dic.update(config)
    args = Namespace(**args_dic)
    return args


if __name__ == '__main__':
    args = build_args()
    # if command line, delete the following two rows; if pycharm debug, add the following two rows
    from config import rct_train_params as params
    args = namespace_add_item_from_dic(args, params)
    result = main(args)
    print('ok')


