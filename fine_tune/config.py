rct_train_params = {
    "mode": 'train+test',
    "num_labels": 2,
    "label_set": ["positive"],
    "dataset": "sentence_data",
    "model_name": "bert_classifier",
    "loss_cal_type": "loss_f",
    "val_list": ["acc", "f1", "eval_subtag"],
    "log_dir": "rct/train_logging",
    "lr": 5e-5,
    "batch_size": 8
}


rct_test_params = {
    "mode": 'test',
    "num_labels": 2,
    "label_set": ["positive"],
    "dataset": "sentence_data",
    "model_name": "bert_classifier",
    "loss_cal_type": "loss_f",
    "val_list": ["acc", "f1", "eval_subtag"],
    "load_dir": "rct/train_logging/version_0/checkpoints",
    # "log_dir": "variation/text_classification/test_logging",
}

