import os
import numpy as np
from fine_tune.reader import ent_set2dic
from collections import defaultdict
# from utils.conlleval import calc_metrics, get_result
# from utils.conlleval import evaluate_bios


def subtag_eval(tag_set, inputs, model_out, only_count=False):
    label_set = ['O']  # 0 to 'O'
    label_set.extend(tag_set)
    tag2idx = {t: i for i, t in enumerate(label_set)}
    unique_labels = list(tag2idx.keys())
    true_seqs = list(inputs['labels'].cpu().numpy())
    logits = model_out["logits"].detach().cpu().numpy()
    pred_seqs = list(np.argmax(logits, axis=1))
    true_seqs = [unique_labels[tag] for tag in true_seqs]
    pred_seqs = [unique_labels[tag] for tag in pred_seqs]
    correct_counts, true_counts, pred_counts = count(true_seqs, pred_seqs, tag_set, count_type="tags")
    correct_counts.pop("O") if "O" in correct_counts else None
    true_counts.pop("O") if "O" in true_counts else None
    pred_counts.pop("O") if "O" in pred_counts else None
    if only_count:
        res = counts2res(correct_counts, true_counts, pred_counts)
        return res
    else:
        res_dict = get_result(correct_counts, true_counts, pred_counts)
        res = {}
        for ent_type, metrics in res_dict.items():
            for metric, value in metrics.items():
                res["_".join([ent_type, metric])] = value
        return res


def entity_eval(ent_set, inputs, model_out, only_count=True):
    tag2idx = ent_set2dic(ent_set)
    unique_labels = list(tag2idx.keys())
    sorted_idx = inputs['sorted_idx']
    org_tok_map = inputs['orig_tok_map']
    original_token = inputs['sents']
    y_true = list(inputs['labels'].cpu().numpy())
    logits = model_out["logits"].detach().cpu().numpy()
    tag_seqs = [list(p) for p in np.argmax(logits, axis=2)]
    writers = []
    for i in range(len(sorted_idx)):
        writer = {"original_token": [], "true_tag": [], "pred_tag": []}
        o2m = org_tok_map[i]
        pos = sorted_idx.index(i)
        for j, orig_tok_idx in enumerate(o2m):
            writer["original_token"].append(original_token[i][j])
            writer["true_tag"].append(unique_labels[y_true[pos][orig_tok_idx]])
            pred_tag = unique_labels[tag_seqs[pos][orig_tok_idx]]
            if pred_tag == 'X':
                pred_tag = 'O'
            writer["pred_tag"].append(pred_tag)
        writers.append(writer)
    if only_count:
        correct_counts, true_counts, pred_counts = evaluate_bios(writers, ent_set, only_count=True)
        res = counts2res(correct_counts, true_counts, pred_counts)
        return res
    else:
        res_dict = evaluate_bios(writers, ent_set)
        res = {}
        for ent_type, metrics in res_dict.items():
            for metric, value in metrics.items():
                res["_".join([ent_type, metric])] = value
        return res


def entity_predict(ent_set, inputs, model_out):
    tag2idx = ent_set2dic(ent_set)
    unique_labels = list(tag2idx.keys())
    sorted_idx = inputs['sorted_idx']
    org_tok_map = inputs['orig_tok_map']
    original_token = inputs['sents']
    # y_true = list(inputs['labels'].cpu().numpy())
    logits = model_out["logits"].detach().cpu().numpy()
    tag_seqs = [list(p) for p in np.argmax(logits, axis=2)]
    tokens = []
    labels = []
    for i in range(len(sorted_idx)):
        token_list = []
        labels_list = []
        o2m = org_tok_map[i]
        pos = sorted_idx.index(i)
        for j, orig_tok_idx in enumerate(o2m):
            token_list.append(original_token[i][j])
            pred_tag = unique_labels[tag_seqs[pos][orig_tok_idx]]
            if pred_tag == 'X':
                pred_tag = 'O'
            labels_list.append(pred_tag)
        tokens.append(token_list)
        labels.append(labels_list)
    return labels


class MetricsCalculator():
    def __init__(self, handshaking_tagger=True):
        self.handshaking_tagger = handshaking_tagger

    def get_ent_cpg(self, pred_ent_list, gold_ent_list):  # set or list
        correct_num, pred_num, gold_num = 0, 0, 0
        gold_ent_set = list(
            ["{}\u2E80{}\u2E80{}".format(ent["char_span"][0], ent["char_span"][1], ent["type"]) for
             ent in gold_ent_list])
        pred_ent_set = list(
            ["{}\u2E80{}\u2E80{}".format(rel["char_span"][0], rel["char_span"][1], rel["type"]) for
             rel in pred_ent_list])
        for rel_str in pred_ent_set:
            if rel_str in gold_ent_set:
                correct_num += 1
        pred_num = len(pred_ent_set)
        gold_num = len(gold_ent_set)
        return correct_num, pred_num, gold_num

    def get_rel_cpg(self, pred_rel_list, gold_rel_list, pattern="whole_text"):  # set or list

        correct_num, pred_num, gold_num = 0, 0, 0

        if pattern == "only_head_index":
            gold_rel_set = list(
                ["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for
                 rel in gold_rel_list])
            pred_rel_set = list(
                ["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for
                 rel in pred_rel_list])
        elif pattern == "whole_span":
            gold_rel_set = list(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                                            rel["subj_tok_span"][1],
                                                                            rel["predicate"],
                                                                            rel["obj_tok_span"][0],
                                                                            rel["obj_tok_span"][1]) for rel in
                                gold_rel_list])
            pred_rel_set = list(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                                            rel["subj_tok_span"][1],
                                                                            rel["predicate"],
                                                                            rel["obj_tok_span"][0],
                                                                            rel["obj_tok_span"][1]) for rel in
                                pred_rel_list])
        elif pattern == "whole_text":
            gold_rel_set = list(
                ["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in
                 gold_rel_list])
            pred_rel_set = list(
                ["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in
                 pred_rel_list])
        elif pattern == "only_head_text":
            gold_rel_set = list(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"],
                                                            rel["object"].split(" ")[0]) for rel in gold_rel_list])
            pred_rel_set = list(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"],
                                                            rel["object"].split(" ")[0]) for rel in pred_rel_list])
        else:
            raise ValueError('pattern error')

        for rel_str in pred_rel_set:
            if rel_str in gold_rel_set:
                correct_num += 1

        pred_num = len(pred_rel_set)
        gold_num = len(gold_rel_set)

        return correct_num, pred_num, gold_num

    def get_ent_cpg4batch(self, pred_ent_batch, gold_ent_batch):
        correct_num, pred_num, gold_num = 0, 0, 0
        for ind in range(len(pred_ent_batch)):
            pred_ent_list = pred_ent_batch[ind]
            gold_ent_list = gold_ent_batch[ind]
            each_correct_num, each_pred_num, each_gold_num = self.get_ent_cpg(pred_ent_list, gold_ent_list)
            correct_num += each_correct_num
            pred_num += each_pred_num
            gold_num += each_gold_num
        return correct_num, pred_num, gold_num

    def get_rel_cpg4batch(self, pred_rel_batch, gold_rel_batch, pattern="whole_text"):
        correct_num, pred_num, gold_num = 0, 0, 0
        for ind in range(len(pred_rel_batch)):
            pred_rel_list = pred_rel_batch[ind]
            gold_rel_list = gold_rel_batch[ind]
            each_correct_num, each_pred_num, each_gold_num = self.get_ent_cpg(pred_rel_list, gold_rel_list)
            correct_num += each_correct_num
            pred_num += each_pred_num
            gold_num += each_gold_num
        return correct_num, pred_num, gold_num

    def get_prf_scores(self, correct_num, pred_num, gold_num, percent=True):
        return calc_metrics(correct_num, pred_num, gold_num, percent=True)


def split_tag(chunk_tag):
    """
    split chunk tag into IOBES prefix and chunk_type
    e.g.
    B-PER -> (B, PER)
    O -> (O, None)
    """
    if chunk_tag == 'O':
        return ('O', None)
    return chunk_tag.split('-', maxsplit=1)


def is_chunk_end(prev_tag, tag):
    """
    check if the previous chunk ended between the previous and current word
    e.g.
    (B-PER, I-PER) -> False
    (B-LOC, O)  -> True

    Note: in case of contradicting tags, e.g. (B-PER, I-LOC)
    this is considered as (B-PER, B-LOC)
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 == 'O':
        return False
    if prefix2 == 'O':
        return prefix1 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']


def is_chunk_start(prev_tag, tag):
    """
    check if a new chunk started between the previous and current word
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 == 'O':
        return False
    if prefix1 == 'O':
        return prefix2 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']


def calc_metrics(tp, p, t, percent=True):
    """
    compute overall precision, recall and FB1 (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    precision = tp / p if p else 0
    recall = tp / t if t else 0
    fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * fb1
    else:
        return precision, recall, fb1


def count(true_seqs, pred_seqs, label_set, count_type="chunks"):
    """
    true_seqs: a list of true tags
    pred_seqs: a list of predicted tags

    return:
    correct_chunks: a dict (counter),
                    key = chunk types,
                    value = number of correctly identified chunks per type
    true_chunks:    a dict, number of true chunks per type
    pred_chunks:    a dict, number of identified chunks per type

    correct_counts, true_counts, pred_counts: similar to above, but for tags
    """
    if count_type == "tags":
        correct_counts = defaultdict(int)
        true_counts = defaultdict(int)
        pred_counts = defaultdict(int)
        # tag_set = list(set([tag for tag in true_seqs + pred_seqs]))
        tag_set = label_set
        for tag in tag_set:
            correct_counts[tag] = 0
            true_counts[tag] = 0
            pred_counts[tag] = 0
        for true_tag, pred_tag in zip(true_seqs, pred_seqs):
            if true_tag == pred_tag and true_tag in tag_set:
                correct_counts[true_tag] += 1
            if true_tag in tag_set:
                true_counts[true_tag] += 1
            if pred_tag in tag_set:
                pred_counts[pred_tag] += 1
        return correct_counts, true_counts, pred_counts
    elif count_type == "chunks":
        correct_chunks = defaultdict(int)
        true_chunks = defaultdict(int)
        pred_chunks = defaultdict(int)

        # type_set = list(set([split_tag(tag)[1] for tag in true_seqs + pred_seqs if split_tag(tag)[1] is not None]))
        type_set = label_set
        for typ in type_set:
            correct_chunks[typ] = 0
            true_chunks[typ] = 0
            pred_chunks[typ] = 0

        prev_true_tag, prev_pred_tag = 'O', 'O'
        correct_chunk = None

        for true_tag, pred_tag in zip(true_seqs, pred_seqs):

            _, true_type = split_tag(true_tag)
            _, pred_type = split_tag(pred_tag)

            if correct_chunk is not None:
                true_end = is_chunk_end(prev_true_tag, true_tag)
                pred_end = is_chunk_end(prev_pred_tag, pred_tag)

                if pred_end and true_end:
                    correct_chunks[correct_chunk] += 1
                    correct_chunk = None
                elif pred_end != true_end or true_type != pred_type:
                    correct_chunk = None

            true_start = is_chunk_start(prev_true_tag, true_tag)
            pred_start = is_chunk_start(prev_pred_tag, pred_tag)

            if true_start and pred_start and true_type == pred_type:
                correct_chunk = true_type
            if true_start:
                true_chunks[true_type] += 1
            if pred_start:
                pred_chunks[pred_type] += 1

            prev_true_tag, prev_pred_tag = true_tag, pred_tag
        if correct_chunk is not None:
            correct_chunks[correct_chunk] += 1
        return correct_chunks, true_chunks, pred_chunks


def get_result(correct_chunks, true_chunks, pred_chunks, verbose=True):
    """
    if verbose, print overall performance, as well as preformance per chunk type;
    otherwise, simply return overall prec, rec, f1 scores
    """
    # sum counts
    sum_correct_chunks = sum(correct_chunks.values())
    sum_true_chunks = sum(true_chunks.values())
    sum_pred_chunks = sum(pred_chunks.values())
    chunk_types = sorted(list(set(list(true_chunks) + list(pred_chunks))))
    prec, rec, f1 = calc_metrics(sum_correct_chunks, sum_pred_chunks, sum_true_chunks)
    res = {"all": {"precision": prec, "recall": rec, "FB1": f1,
                          "true": sum_true_chunks, "pred": sum_pred_chunks, "correct": sum_correct_chunks}}
    if not verbose:
        return res

    # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
    for t in chunk_types:
        prec, rec, f1 = calc_metrics(correct_chunks[t], pred_chunks[t], true_chunks[t])
        res[t] = {"precision": prec, "recall": rec, "FB1": f1, "true": true_chunks[t],
                  "pred": pred_chunks[t], "correct": correct_chunks[t]}
    return res


def counts2res(correct_counts, true_counts, pred_counts):
    res = {}
    res.update({key + "_correct": value for key, value in correct_counts.items()})
    res.update({key + "_true": value for key, value in true_counts.items()})
    res.update({key + "_pred": value for key, value in pred_counts.items()})
    res.update({"all_correct": sum(correct_counts.values())})
    res.update({"all_true": sum(true_counts.values())})
    res.update({"all_pred": sum(pred_counts.values())})
    return res


# writers: [{"original_token": [], "true_tag": [], "pred_tag": []},~]
def evaluate_bios(writers, ent_set, verbose=True, only_count=False):
    true_seqs, pred_seqs = [], []
    for writer in writers:
        true_seqs += writer["true_tag"]
        pred_seqs += writer["pred_tag"]
    if only_count:
        correct_chunks, true_chunks, pred_chunks = count(true_seqs, pred_seqs, ent_set, count_type="chunks")
        return correct_chunks, true_chunks, pred_chunks
    else:
        correct_chunks, true_chunks, pred_chunks = count(true_seqs, pred_seqs, ent_set, count_type="chunks")
        result = get_result(correct_chunks, true_chunks, pred_chunks, verbose=verbose)
        return result



