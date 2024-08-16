import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
from collections import OrderedDict
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1, recall, precision, fbeta, confusion_matrix
import torch.nn as nn
from torch import optim
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import Counter
from functools import reduce
from typing import Any, Optional
import sys
sys.path.append(sys.path[0]+'/../../..')
from fine_tune.metrics import subtag_eval, entity_eval, entity_predict, calc_metrics


class MInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()   # Save model arguments to ``hparams`` attribute
        self.load_model()
        self.configure_loss()
        if hasattr(self.model, 'nr_frozen_epochs') and self.model.nr_frozen_epochs > 0:
            self.model.freeze_encoder()
        else:
            self.model._frozen = False

    # def forward(self, tokens, lengths):
    #     return self.model(tokens, lengths)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        # model_out = self(**inputs)
        model_out = self.model(**inputs)
        if self.hparams.loss_cal_type == 'loss_f':
            loss = self.loss_function(model_out["logits"], targets["labels"])
        elif self.hparams.loss_cal_type == 'model_out':
            loss = model_out['loss']
        elif self.hparams.loss_cal_type == 'model_f':
            loss = self.model.loss_function(**inputs)
        output = OrderedDict(
            {"loss": loss}
        )
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return output

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        # model_out = self(**inputs)
        model_out = self.model(**inputs)
        # loss
        if self.hparams.loss_cal_type == 'loss_f':
            loss_val = self.loss_function(model_out["logits"], targets["labels"])
        elif self.hparams.loss_cal_type == 'model_out':
            loss_val = model_out['loss']
        elif self.hparams.loss_cal_type == 'model_f':
            loss_val = self.model.loss_function(**inputs)
        else:
            return ValueError

        output = OrderedDict({"val_loss": loss_val})

        # validation
        if 'eval_entity' in self.hparams.val_list:
            res = entity_eval(self.hparams.label_set, inputs, model_out, only_count=True)
            output.update(res)
        elif 'eval_subtag' in self.hparams.val_list:
            res = subtag_eval(self.hparams.label_set, targets, model_out, only_count=True)
            output.update(res)
        else:
            return ValueError

        for key, value in output.items():
            value = torch.tensor(value).to(torch.device("cuda", 0))
            output[key] = value + 1e-6  # if len(x["key"])=0, Counter(x) del "key"
            # self.log(key, value, on_step=False, on_epoch=True, prog_bar=True)
        return output

    def validation_epoch_end(self, outputs: list) -> dict:
        total = reduce(lambda x, y: Counter(x) + Counter(y), outputs)
        tqdm = {}
        for k, v in total.items():
            if "pred" in k or "true" in k or "correct" in k:
                tqdm['epoch_' + k] = v
            else:
                tqdm['epoch_' + k] = v / len(outputs)
        tag_set = {"all"}
        tag_set.update(self.hparams.label_set)
        print('\nversion: ' + str(self.logger.version) + '  epoch: ' + str(self.current_epoch))
        for tag in sorted(tag_set):
            tqdm['epoch_'+tag+'_precision'], tqdm['epoch_'+tag+'_recall'], tqdm['epoch_'+tag+'_f1'] = \
                calc_metrics(tqdm['epoch_'+tag+'_correct'], tqdm['epoch_'+tag+'_pred'], tqdm['epoch_'+tag+'_true'])
            print("%s:   precision: %.4f; recall: %.4f; FB1: %.4f; %i %i %i"
                  % (tag, tqdm['epoch_'+tag+'_precision'], tqdm['epoch_'+tag+'_recall'], tqdm['epoch_'+tag+'_f1'],
                     tqdm['epoch_'+tag+'_correct'], tqdm['epoch_'+tag+'_pred'], tqdm['epoch_'+tag+'_true']))
        # self.log only save one digit
        self.log('version', self.logger.version, on_step=False, on_epoch=True, prog_bar=True)
        self.log('epoch', self.current_epoch, on_step=False, on_epoch=True, prog_bar=True)
        self.log('epoch_val_loss', tqdm["epoch_val_loss"], on_step=False, on_epoch=True, prog_bar=True)
        if 'eval_entity' in self.hparams.val_list or 'eval_subtag' in self.hparams.val_list:
            self.log('epoch_all_precision', tqdm["epoch_all_precision"],
                     on_step=False, on_epoch=True, prog_bar=True)
            self.log('epoch_all_recall', tqdm["epoch_all_recall"],
                     on_step=False, on_epoch=True, prog_bar=True)
            self.log('epoch_all_f1', tqdm["epoch_all_f1"],
                     on_step=False, on_epoch=True, prog_bar=True)
        # print('\nepoch_end:')
        # print({key: value.cpu().item() for key, value in tqdm.items()})
        # print('\n')
        result = tqdm
        return result

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs: list) -> dict:
        return self.validation_epoch_end(outputs)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        if self.hparams.loss_cal_type == 'model_out':
            inputs, targets = batch
            model_out = self.model(**inputs)
            labels_hat = entity_predict(self.hparams.label_set, inputs, model_out)
        else:
            try:
                model_out = self.model(**batch)
            except:
                inputs, targets = batch
                model_out = self.model(**inputs)
            labels_hat = torch.argmax(model_out["logits"], dim=1)
        return labels_hat

    def predict(self, inputs):
        model_out = self.model(**inputs)
        if self.hparams.loss_cal_type == 'model_out':
            labels_hat = entity_predict(self.hparams.label_set, inputs, model_out)
        else:
            labels_hat = torch.argmax(model_out["logits"], dim=1)
        return labels_hat

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if hasattr(self.model, 'nr_frozen_epochs'):
            if self.current_epoch + 1 > self.model.nr_frozen_epochs:
                self.model.unfreeze_encoder()

    def configure_optimizers(self):
        # weight_decay
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        # parameters
        if hasattr(self.model, 'train_parameters'):
            parameters = self.model.train_parameters
        else:
            parameters = self.parameters()
        # optimizer
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                parameters, lr=self.hparams.lr, weight_decay=weight_decay, eps=self.hparams.eps)
        elif self.hparams.optimizer == 'adamw':
            optimizer = AdamW(parameters, lr=self.hparams.lr, weight_decay=weight_decay, eps=self.hparams.eps)
        else:
            raise ValueError('Invalid optimizer type!')
        # scheduler
        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            elif self.hparams.lr_scheduler == 'warmup':
                scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=2,
                                                            num_training_steps=self.hparams.batch_size * self.hparams.max_epochs)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss()
        elif loss == 'l1':
            self.loss_function = F.l1_loss()
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy()
        elif loss == 'ce':
            self.loss_function = nn.CrossEntropyLoss()
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module('.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        if self.hparams.is_pretrain_model:
            class_args = self.hparams.pretrain_model_parameter
            class_args.extend(inspect.getargspec(Model.__init__).args[1:])
        else:
            class_args = inspect.getargspec(Model.__init__).args[1:]

        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)

        if self.hparams.is_pretrain_model:
            return Model.from_pretrained(**args1)
        else:
            return Model(**args1)
