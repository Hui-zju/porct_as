import torch
import numpy as np
from torch import nn
from transformers import AutoModel
from transformers import AutoTokenizer
import logging as log


class BertClassifier(nn.Module):
    def __init__(self, encoder_lr, nr_frozen_epochs, num_labels, pretrained_model_name_or_path):
        super().__init__()
        self.encoder_lr = encoder_lr
        self.nr_frozen_epochs = nr_frozen_epochs
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        # self.encoder_features = 128
        self.encoder_features = 768
        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features * 2),
            nn.Tanh(),
            nn.Linear(self.encoder_features * 2, self.encoder_features),
            nn.Tanh(),
            nn.Linear(self.encoder_features, self.num_labels),
        )

        self.train_parameters = [
            {"params": self.classification_head.parameters()},
            {
                "params": self.bert.parameters(),
                "lr": self.encoder_lr,
            },
        ]

    def forward(self, **kwargs):
        tokens = kwargs['tokens']
        lengths = kwargs['lengths']
        tokens = tokens[:, : lengths.max()]
        mask = lengths_to_mask(lengths, device=tokens.device)

        # Run BERT model.
        word_embeddings = self.bert(tokens, mask)[0]

        # Average Pooling
        word_embeddings = mask_fill(
            0.0, tokens, word_embeddings, self.tokenizer.pad_token_id
        )
        sentemb = torch.sum(word_embeddings, 1)
        sum_mask = mask.unsqueeze(-1).expand(word_embeddings.size()).float().sum(1)
        sentemb = sentemb / sum_mask
        return {"logits": self.classification_head(sentemb)}

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.bert.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.bert.parameters():
            param.requires_grad = False
        self._frozen = True


def mask_fill(
        fill_value: float,
        tokens: torch.tensor,
        embeddings: torch.tensor,
        padding_index: int,
) -> torch.tensor:
    """
    Function that masks embeddings representing padded elements.
    :param fill_value: the value to fill the embeddings belonging to padded tokens.
    :param tokens: The input sequences [bsz x seq_len].
    :param embeddings: word embeddings [bsz x seq_len x hiddens].
    :param padding_index: Index of the padding token.
    """
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)


def lengths_to_mask(*lengths, **kwargs):
    """ Given a list of lengths, create a batch mask.

    Example:
        >>> lengths_to_mask([1, 2, 3])
        tensor([[ True, False, False],
                [ True,  True, False],
                [ True,  True,  True]])
        >>> lengths_to_mask([1, 2, 2], [1, 2, 2])
        tensor([[[ True, False],
                 [False, False]],
        <BLANKLINE>
                [[ True,  True],
                 [ True,  True]],
        <BLANKLINE>
                [[ True,  True],
                 [ True,  True]]])

    Args:
        *lengths (list of int or torch.Tensor)
        **kwargs: Keyword arguments passed to ``torch.zeros`` upon initially creating the returned
          tensor.

    Returns:
        torch.BoolTensor
    """
    # Squeeze to deal with random additional dimensions
    lengths = [l.squeeze().tolist() if torch.is_tensor(l) else l for l in lengths]

    # For cases where length is a scalar, this needs to convert it to a list.
    lengths = [l if isinstance(l, list) else [l] for l in lengths]
    assert all(len(l) == len(lengths[0]) for l in lengths)
    batch_size = len(lengths[0])
    other_dimensions = tuple([int(max(l)) for l in lengths])
    mask = torch.zeros(batch_size, *other_dimensions, **kwargs)
    for i, length in enumerate(zip(*tuple(lengths))):
        mask[i][[slice(int(l)) for l in length]].fill_(1)
    return mask.bool()
