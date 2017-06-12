#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
from __future__ import print_function
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import torch
import onmt

parser = argparse.ArgumentParser(description='embeddings_to_torch.py')

##
## **Preprocess Options**
##

parser.add_argument('-emb_file', required=True, help="Embeddings from this file")
parser.add_argument('-output_file', required=True,
                    help="Output file for the prepared data")
parser.add_argument('-dict_file', required=True,
                    help="Dictionary file")
opt = parser.parse_args()


def get_vocabs(dict_file):
    vocab = dict()
    line_num = 1
    for l in open(dict_file).readlines():
        l_split = l.decode('utf8').rstrip().split()
        if len(l_split) != 2:
            print ("Line %s error" % line_num)
        else:
            if l_split[0] in vocab:
                print(l_split[0].encode('utf8'), ' duplicate word.')
            vocab[l_split[0]] = int(l_split[1])
        line_num += 1
    print("Got {} words from {}".format(len(vocab), dict_file))

    return vocab

def get_embeddings(file):
    embs = dict()
    for l in open(file).readlines():
        l_split = l.decode('utf8').strip().split()
        if len(l_split) == 2:
            continue
        embs[l_split[0]] = [float(em) for em in l_split[1:]]
    print("Got {} embeddings from {}".format(len(embs), file))

    return embs

def match_embeddings(vocab, emb):
    filtered_embeddings = np.zeros((len(vocab), len(emb.itervalues().next())))
    for w, w_id in vocab.iteritems():
        if w in emb:
            filtered_embeddings[w_id] = emb[w]
        else:
            print(u"{} not found".format(w))

    return torch.Tensor(filtered_embeddings)

def main():
    vocabs = get_vocabs(opt.dict_file)
    embeddings = get_embeddings(opt.emb_file)
    filtered_embeddings = match_embeddings(vocabs, embeddings)
    torch.save(filtered_embeddings, opt.output_file)
    print(filtered_embeddings.size())

if __name__ == "__main__":
    main()
