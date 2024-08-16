import os
import gc
import json
import torch
import shutil
from main import build_args, namespace_add_item_from_dic, main


def train(data_path, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    results = {}
    test_path = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path) if "test" in file_name][0]
    for file_name in os.listdir(data_path):
        args = build_args()
        if "train" in file_name:
            train_path = os.path.join(data_path, file_name)
            val_path = os.path.join(data_path, file_name.replace("train", "val"))
            from config import rct_train_params as params
            params["train_data_dir"] = train_path
            params["val_data_dir"] = val_path
            params["test_data_dir"] = test_path
            print("\n\n\n")
            print(train_path)
            print("\n\n\n")
            params["log_dir"] = os.path.join(save_folder, file_name[:-4])
            args = namespace_add_item_from_dic(args, params)
            result = main(args)
            results[file_name[6:-4]] = result
            gc.collect()
            torch.cuda.empty_cache()
    save_path = os.path.join(save_folder, "result.json")
    with open(save_path, 'w') as file:
        json.dump(results, file)


def test(data_path, model_folder, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    results = {}
    test_path = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path) if "test" in file_name][0]
    for file_name in os.listdir(model_folder):
        args = build_args()
        if "train" in file_name:
            from config import rct_test_params as params
            params["test_data_dir"] = test_path
            params["load_dir"] = os.path.join(model_folder, file_name, "version_0/checkpoints/best_model.ckpt")
            args = namespace_add_item_from_dic(args, params)
            result = main(args)
            results[file_name[6:]] = result
            gc.collect()
            torch.cuda.empty_cache()
    save_path = os.path.join(save_folder, "result.json")
    with open(save_path, 'w') as file:
        json.dump(results, file)


if __name__ == '__main__':
    data_path = "../dataset/fine_tune_dataset/collaborative_annotation"
    save_folder = "result/fine_tune/collaborative_annotation/train_logging_1"
    train(data_path, save_folder)

    # model_folder = "../result/fine_tune/collaborative_annotation/train_logging"
    # save_folder = "../result/fine_tune/collaborative_annotation/test_logging"
    # test(data_path,  model_folder, save_folder)
    #
    #
    # data_path = "../dataset/fine_tune_dataset/llm_annotation"
    # save_folder = "result/fine_tune/llm_annotation/train_logging"
    # train(data_path, save_folder)
    #
    # model_folder = "../result/fine_tune/llm_annotation/train_logging"
    # save_folder = "../result/fine_tune/llm_annotation/test_logging"
    # test(data_path,  model_folder, save_folder)






