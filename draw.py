from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import json

from method.llm_api.compute_metric import get_pred_label


def get_metric(json_path):
    with open(json_path, 'r') as file:
        result = json.load(file)
    values = result.values()
    precision = [item[0]['epoch_all_precision']/100 for item in values][::2]
    recall = [item[0]['epoch_all_recall']/100 for item in values][::2]
    f1 = [item[0]['epoch_all_f1']/100 for item in values][::2]
    return precision, recall, f1


def count_false_pos(true_label, pred_label):
    return len([idx for idx in range(len(true_label)) if true_label[idx] == 0 and pred_label[idx] == 1])


def count_true_pos(true_label, pred_label):
    return len([idx for idx in range(len(true_label)) if true_label[idx] == 1 and pred_label[idx] == 1])


if __name__ == '__main__':
    # fig, ax1 = plt.subplots(figsize=(170/25.4, 8))
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(170 / 25.4, 70 / 25.4))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.15, wspace=0.2, hspace=0.5)


    annotation_path = "dataset/api_dataset/collaborative_annotation_1800.csv"
    sen_data = pd.read_csv(annotation_path).to_dict("records")  # , sep='\t'
    collaborative_label = [each['label'] for idx, each in enumerate(sen_data)]

    predict_path = "result/api/final_prompt/collaborative_annotation_fine_tune_1800.json"
    with open(predict_path, 'r') as file:
        response_labels = json.load(file)
    return_confidence = False
    llm_label = get_pred_label(response_labels, return_confidence)

    num_iterations = len(collaborative_label) + 1
    batch_size = 20
    label_counts = np.arange(batch_size, num_iterations, batch_size)
    pred_neg_counts = [item - sum(llm_label[:item]) for item in label_counts]
    pred_pos_counts = [sum(llm_label[:item]) for item in label_counts]
    collaborative_neg_counts = [item - sum(collaborative_label[:item]) for item in label_counts]
    collaborative_pos_counts = [sum(collaborative_label[:item]) for item in label_counts]

    false_pos_counts = [count_false_pos(collaborative_label[:item], llm_label[:item]) for item in label_counts]
    true_pos_counts = [count_true_pos(collaborative_label[:item], llm_label[:item]) for item in label_counts]

    ax1.plot(label_counts, pred_pos_counts, label='positive labels by LLM-only', linewidth=1.5)
    ax1.plot(label_counts, pred_neg_counts, label='negative labels by LLM-only', linewidth=1.5)
    ax1.plot(label_counts, collaborative_pos_counts, label='positive labels by Human-LLM', linewidth=1.5)
    ax1.plot(label_counts, collaborative_neg_counts, label='negative labels by Human-LLM', linewidth=1.5)
    ax1.set_xlabel('Number of articles', fontsize=6)
    ax1.set_ylabel('Number of labels', fontsize=6)
    ax1.set_xlim(0, 1800)
    ax1.set_ylim(0, 1800)
    ax1.legend(loc='upper right', fontsize=4)
    ax1.tick_params(axis='both', which='major', labelsize=6)
    ax1.set_title('(a)', fontsize=6)

    col_json_path = "result/fine_tune/collaborative_annotation/test_logging/result.json"
    col_precision, col_recall, col_f1 = get_metric(col_json_path)
    llm_json_path = "result/fine_tune/llm_annotation/test_logging/result.json"
    llm_precision, llm_recall, llm_f1 = get_metric(llm_json_path)

    x = range(200, 2001, 400)
    ax2.plot(x, col_precision, color='#1f77b4', linestyle='-')
    ax2.plot(x, col_recall, color='#003366', linestyle='-')
    ax2.plot(x, col_f1, color='#2ca02c', linestyle='-')

    ax2.plot(x, llm_precision, color='#1f77b4', linestyle='--')
    ax2.plot(x, llm_recall, color='#003366', linestyle='--')
    ax2.plot(x, llm_f1, color='#2ca02c', linestyle='--')

    ax2.axhline(y=0.4615, color='#1f77b4', linestyle=':')  # ?????
    ax2.axhline(y=0.8571, color='#003366', linestyle=':')  # ?????
    ax2.axhline(y=0.6000, color='#2ca02c', linestyle=':')

    legend_patches = [
        Patch(color='#1f77b4', label='Precision'),
        Patch(color='#003366', label='Recall'),
        Patch(color='#2ca02c', label='F1'),
        Line2D([0], [0], color='black', linestyle='-', linewidth=1,
               label='Model on collaborative-data'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1, label='Model on LLM-data'),
        Line2D([0], [0], color='black', linestyle=':', linewidth=1, label='LLM annotation threshold')
    ]
    ax2.set_xlabel('The amount of training data', fontsize=6)
    ax2.set_ylabel('Metric', fontsize=6)
    ax2.legend(handles=legend_patches, loc='upper right', fontsize=4)
    ax2.tick_params(axis='both', which='major', labelsize=6)
    ax2.set_title('(b)', fontsize=6)
    plt.savefig('result/annotation.tiff', dpi=300, format='tiff')


    print("ok")


