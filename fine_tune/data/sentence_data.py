import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split, KFold
from transformers import AutoTokenizer
from fine_tune.tools import collate_tensors


class SentenceData(data.Dataset):
    # data_dir VS train_data_dir + val_data_dir + test_data_dir
    def __init__(self,
                 train_data_dir=None,
                 val_data_dir=None,
                 test_data_dir=None,
                 predict_data_dir=None,
                 predict_data=None,
                 datatype=None,
                 kfold=0,
                 fold_num=0,
                 pretrained_model_name_or_path="bert-base-uncased",
                 vec_max_len=80,):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.check_files()

    def check_files(self):
        if self.train_data_dir is not None:
            train_sen_data = pd.read_csv(self.train_data_dir).to_dict("records")
            if self.val_data_dir is not None:
                val_sen_data = pd.read_csv(self.val_data_dir).to_dict("records")
            elif self.kfold != 0:
                kf = KFold(n_splits=self.kfold, shuffle=True, random_state=2333)
                train_idx, val_idx = list(kf.split(train_sen_data))[self.fold_num]
                val_sen_data = np.array(train_sen_data)[val_idx]  # watch the order
                train_sen_data = np.array(train_sen_data)[train_idx]
            else:
                train_sen_data, val_sen_data = train_test_split(
                    train_sen_data, test_size=0.15, random_state=2333)
        if self.test_data_dir is not None:
            test_sen_data = pd.read_csv(self.test_data_dir).to_dict("records")
        if self.predict_data_dir is not None:
            predict_sen_data = pd.read_csv(self.predict_data_dir).to_dict("records")
        if self.predict_data is not None:
            predict_sen_data = self.predict_data

        if self.datatype == 'train':
            self.path_list = train_sen_data
        elif self.datatype == 'val':
            self.path_list = val_sen_data
        elif self.datatype == 'test':
            self.path_list = test_sen_data
        elif self.datatype == 'predict':
            self.path_list = predict_sen_data
        else:
            raise ValueError("datatype should be 'train' , 'val', 'predict' or 'test'.")

    def collate_fn(self, sample) -> (torch.Tensor, torch.Tensor):
        sample = collate_tensors(sample)
        sentences = sample['sentence']
        tokenizer_output = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            return_length=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            truncation="only_first",
            max_length=self.vec_max_len
        )
        tokens = tokenizer_output["input_ids"]
        lengths = tokenizer_output["length"]
        inputs = {"tokens": tokens, "lengths": lengths}
        if 'label' not in sample.keys():
            return inputs
        else:
            targets = {"labels": torch.Tensor(list(map(int, sample['label']))).long()}
            return inputs, targets

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        sen_data = self.path_list[index]
        return sen_data


if __name__ == "__main__":
    dataset = SentenceData(data_dir='./sen_data/variation_sentence_classification_training_data.csv', datatype='train')
    train_iter = data.DataLoader(dataset=dataset,
                                 batch_size=4,
                                 collate_fn=dataset.collate_fn
                                 )
    for sen in train_iter:
        print(sen)








