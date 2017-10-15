#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
from __future__ import print_function
from __future__ import division
import argparse
import collections
import six
import numpy as np

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-ans', required=True,
                    help='Path to answers')
parser.add_argument('-pred', required=True,
                    help='Path to predictions')
parser.add_argument('-output_ranked_r2',
                    help='Path to write predictions ranked by R-2.')
parser.add_argument('-source',
                    help='If output ranked prediction, this path points to the input.')
opt = parser.parse_args()

def _ngrams(words, n):
    queue = collections.deque(maxlen=n)
    for w in words:
        queue.append(w)
        if len(queue) == n:
            yield tuple(queue)

def _ngram_counts(words, n):
    return collections.Counter(_ngrams(words, n))

def _ngram_count(words, n):
    return max(len(words) - n + 1, 0)

def _counter_overlap(counter1, counter2):
    result = 0
    for k, v in six.iteritems(counter1):
        result += min(v, counter2[k])
    return result

def _safe_divide(numerator, denominator):
    if denominator > 0:
        return numerator / denominator
    else:
        return 0

def _safe_f1(matches, recall_total, precision_total, alpha=1):
    recall_score = _safe_divide(matches, recall_total)
    precision_score = _safe_divide(matches, precision_total)
    denom = (1.0 - alpha) * precision_score + alpha * recall_score
    if denom > 0.0:
        return (precision_score * recall_score) / denom
    else:
        return 0.0

def rouge_n(peer, models, n, alpha=1):
    """
    Compute the ROUGE-N score of a peer with respect to one or more models, for
    a given value of `n`.
    """
    matches = 0
    recall_total = 0
    peer_counter = _ngram_counts(peer, n)
    for model in models:
        model_counter = _ngram_counts(model, n)
        matches += _counter_overlap(peer_counter, model_counter)
        recall_total += _ngram_count(model, n)
    precision_total = len(models) * _ngram_count(peer, n)
    return _safe_f1(matches, recall_total, precision_total, alpha)

def rouge_1(peer, models, alpha=1):
    """
    Compute the ROUGE-1 (unigram) score of a peer with respect to one or more
    models.
    """
    return rouge_n(peer, models, 1, alpha)

def rouge_2(peer, models, alpha=1):
    """
    Compute the ROUGE-2 (bigram) score of a peer with respect to one or more
    models.
    """
    return rouge_n(peer, models, 2, alpha)

def rouge_3(peer, models, alpha=1):
    """
    Compute the ROUGE-3 (trigram) score of a peer with respect to one or more
    models.
    """
    return rouge_n(peer, models, 3, alpha)

def lcs(a, b):
    """
    Compute the length of the longest common subsequence between two sequences.

    Time complexity: O(len(a) * len(b))
    Space complexity: O(min(len(a), len(b)))
    """
    # This is an adaptation of the standard LCS dynamic programming algorithm
    # tweaked for lower memory consumption.
    # Sequence a is laid out along the rows, b along the columns.
    # Minimize number of columns to minimize required memory
    if len(a) < len(b):
        a, b = b, a
    # Sequence b now has the minimum length
    # Quit early if one sequence is empty
    if len(b) == 0:
        return 0
    # Use a single buffer to store the counts for the current row, and
    # overwrite it on each pass
    row = [0] * len(b)
    for ai in a:
        left = 0
        diag = 0
        for j, bj in enumerate(b):
            up = row[j]
            if ai == bj:
                value = diag + 1
            else:
                value = max(left, up)
            row[j] = value
            left = value
            diag = up
    # Return the last cell of the last row
    return left

def rouge_l(peer, models, alpha=1):
    """
    Compute the ROUGE-L score of a peer with respect to one or more models.
    """
    matches = 0
    recall_total = 0
    for model in models:
        matches += lcs(model, peer)
        recall_total += len(model)
    precision_total = len(models) * len(peer)
    return _safe_f1(matches, recall_total, precision_total, alpha)

if __name__ == "__main__":
    answers = []
    for line in open(opt.ans).readlines():
        line_split = line.decode('utf8').strip()
        answers.append([line_split])
    predictions = []
    for line in open(opt.pred).readlines():
        line_split = line.decode('utf8').strip()
        predictions.append(line_split)
    rouge_ones = []
    rouge_twos = []
    rouge_ls = []
    for pred_id in range(len(predictions)):
        rouge_ones.append(rouge_1(predictions[pred_id], answers[pred_id]))
        rouge_twos.append(rouge_2(predictions[pred_id], answers[pred_id]))
        rouge_ls.append(rouge_l(predictions[pred_id], answers[pred_id]))
    print("R-1: {:.4f}".format(sum(rouge_ones)/len(predictions)))
    print("R-2: {:.4f}".format(sum(rouge_twos)/len(predictions)))
    print("R-L: {:.4f}".format(sum(rouge_ls)/len(predictions)))
    if not opt.output_ranked_r2: exit()
    if opt.source:
        sources = []
        for line in open(opt.source).readlines():
            line_split = line.decode('utf8').strip().split('\t')
            sources.append(line_split[0])
    ranked_ids = np.argsort(rouge_twos)[::-1]
    with open(opt.output_ranked_r2, 'w') as ranked_pred_file:
        ranked_pred_file.write("l_id\trouge_2F\tpred\tgold\torg\n")
        for ranked_id in ranked_ids:
            ranked_pred_file.write("{}".format(ranked_ids[ranked_id]+1))
            ranked_pred_file.write('\t')
            ranked_pred_file.write("{:.4f}".format(rouge_twos[ranked_id]))
            ranked_pred_file.write('\t')
            ranked_pred_file.write((''.join(predictions[ranked_id])).encode('utf8'))
            ranked_pred_file.write('\t')
            ranked_pred_file.write((''.join(answers[ranked_id][0])).encode('utf8'))
            ranked_pred_file.write('\t')
            if opt.source is not None:
                ranked_pred_file.write((''.join(sources[ranked_id])).encode('utf8'))
            else:
                ranked_pred_file.write(u'-')
            ranked_pred_file.write('\n')
