
import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DInterface(pl.LightningDataModule):
    def __init__(self, num_workers=8,
                 dataset='',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.load_data_module()

    # 该函数在trainer.py中被使用
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(datatype='train')
            self.valset = self.instancialize(datatype='val')

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(datatype='test')

        if stage == 'predict' or stage is None:
            self.predictset = self.instancialize(datatype='predict')


    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.batch_size,
                          collate_fn=self.trainset.collate_fn,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset,
                          batch_size=self.batch_size,
                          collate_fn=self.valset.collate_fn,
                          num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset,
                          batch_size=self.batch_size,
                          collate_fn=self.testset.collate_fn,
                          num_workers=self.num_workers,
                          shuffle=False)

    def predict_dataloader(self, predict_data=None):
        if predict_data is not None:
            self.predictset = self.instancialize(datatype='predict', predict_data=predict_data)
        return DataLoader(self.predictset,
                          batch_size=self.batch_size,
                          collate_fn=self.predictset.collate_fn,
                          num_workers=self.num_workers,
                          shuffle=False)


    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)
