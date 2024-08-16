import json
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from process_response import get_label_from_response, repair_json_by_chatgpt


def compute_metrics(labels, preds):
    matrix = confusion_matrix(labels, preds, labels=[0, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(labels.flatten(), preds.flatten(), labels=[0, 1], zero_division=0)
    return {
        'f1': f1[1],
        'precision': precision[1],
        'recall': recall[1],
        'true_num': np.sum(matrix, 1)[1],
        'pred_num': np.sum(matrix, 0)[1],
        'correct_num': np.diag(matrix)[1],
    }


def get_pred_label(response_labels, return_confidence = False):
    preds = []
    for idy, response_label in enumerate(response_labels):
        pred = []
        for idx, item in enumerate(response_label):
            raw_label = item["response"]
            label = get_label_from_response(raw_label)
            if return_confidence:
                confidence = item["confidence"]
                pred.append({"label": label, "confidence": confidence})
            else:
                pred.append({"label": label})
        preds.append(pred)
    pred_label = [1 if all(element["label"] == 1 for element in pred) else 0 for pred in preds]
    return pred_label


if __name__ == '__main__':
    true_path = "../dataset/api_dataset/prompt_optimization_tuning_set_100.csv"
    sen_data = pd.read_csv(true_path).to_dict("records")
    true_label = [each['label'] for idx, each in enumerate(sen_data)]

    predict_path = "../result/prompt_optimization/final_prompt/prompt_optimization_tuning_set_100_gpt-3.5-turbo_0_single1_.json"
    with open(predict_path, 'r') as file:
        response_labels = json.load(file)
    return_confidence = False
    pred_label = get_pred_label(response_labels, return_confidence)
    metrics = compute_metrics(np.array(true_label), np.array(pred_label))

    # pmid = [each['pmid'] for _, each in enumerate(sen_data)]
    # positive_pmid = [each['pmid'] for idx, each in enumerate(sen_data) if each['label'] == 1]
    # true_positive_idx = [idx for idx, item in enumerate(pred_label) if item == 1 and true_label[idx] == 1]
    # false_positive_idx = [idx for idx, item in enumerate(pred_label) if item == 1 and true_label[idx] == 0]
    # false_negative_idx = [idx for idx, item in enumerate(pred_label) if item == 0 and true_label[idx] == 1]
    # true_positive_pmid = [item for idx, item in enumerate(pmid) if idx in true_positive_idx]
    # false_positive_pmid = [item for idx, item in enumerate(pmid) if idx in false_positive_idx]
    # false_negative_pmid = [item for idx, item in enumerate(pmid) if idx in false_negative_idx]
    # true_positive_label = [item for idx, item in enumerate(response_labels) if idx in true_positive_idx]
    # false_negative_label = [item for idx, item in enumerate(response_labels) if idx in false_negative_idx]
    # false_positive_label = [item for idx, item in enumerate(response_labels) if idx in false_positive_idx]
    print('ok')


