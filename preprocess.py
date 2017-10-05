#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------

from __future__ import print_function
from __future__ import division
import matplotlib
import onmt
import argparse
import torch
import re

matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='preprocess.py')

##
## **Preprocess Options**
##

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-train_file', required=True,
                    help="Path to the training data")
# parser.add_argument('-valid_file', required=True,
#                     help="Path to the validation data")
parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=4000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=4000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")

parser.add_argument('-max_seq_length', type=int, default=120,
                    help="Maximum sequence length")
parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=1337,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=1000,
                    help="Report status every this many sentences")

opt = parser.parse_args()


def makeVocabulary(lines, size):
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                       lower=opt.lower)

    sent_num = 1
    for sent in lines:
        if sent is not list:
            sent = list(sent)
        for word in sent:
            vocab.add(word)
        sent_num += 1

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(dataFile, src_vocab_file, src_vocabSize, tgt_vocab_file, tgt_vocabSize):

    src_vocab = None
    tgt_vocab = None
    if src_vocab_file is not None:
        # If given, load existing word dictionary.
        print('Reading vocabulary from \'' + src_vocab_file + '\'...')
        src_vocab = onmt.Dict()
        src_vocab.loadFile(src_vocab_file)
        print('Loaded ' + str(src_vocab.size()) + ' source words')
    if tgt_vocab_file is not None:
        # If given, load existing word dictionary.
        print('Reading vocabulary from \'' + tgt_vocab_file + '\'...')
        tgt_vocab = onmt.Dict()
        tgt_vocab.loadFile(tgt_vocab_file)
        print('Loaded ' + str(tgt_vocab.size()) + ' target words')
    if src_vocab and tgt_vocab:
        # early return
        return src_vocab, tgt_vocab

    with open(dataFile) as dataFileHandle:
        lines_in_file = [l.strip().decode('utf8').split('\t') for l in dataFileHandle.readlines()]
    line_num = 1
    for l in lines_in_file:
        if len(l) != 2:
            print("Error on line %d" % (line_num))
        line_num += 1
    if src_vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building source vocabulary...')
        src_lines = [l[0] for l in lines_in_file]
        src_vocab = makeVocabulary(src_lines, src_vocabSize)

    if tgt_vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building target vocabulary...')
        tgt_lines = [l[1] for l in lines_in_file]
        tgt_vocab = makeVocabulary(tgt_lines, tgt_vocabSize)

    return src_vocab, tgt_vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(filename, srcDict, tgtDict):
    src, tgt = [], []
    sizes = []
    count, truncated = 0, 0

    print('Processing file %s...' % (filename))
    with open(filename) as inputFile:
        for src_and_tgt_line in inputFile.readlines():
            if len(src_and_tgt_line) < 1:
                continue
            src_and_tgt_line = src_and_tgt_line.strip().decode('utf8').split('\t')
            if len(src_and_tgt_line) != 2:
                print("Error on line %d" % (count + 1))
                continue
            srcWords = re.sub(' +', '', src_and_tgt_line[0])
            tgtWords = re.sub(' +', '', src_and_tgt_line[1])
            srcWords = list(srcWords)
            tgtWords = list(tgtWords)

            if (len(srcWords) <= opt.max_seq_length) and \
               (len(tgtWords) <= opt.max_seq_length):

                src += [srcDict.convertToIdx(labels=srcWords,
                                             unkWord=onmt.Constants.UNK_WORD)]
                tgt += [tgtDict.convertToIdx(labels=tgtWords,
                                             unkWord=onmt.Constants.UNK_WORD,
                                             bosWord=onmt.Constants.BOS_WORD,
                                             eosWord=onmt.Constants.EOS_WORD)]
                sizes += [len(srcWords)]
            else:
                src += [srcDict.convertToIdx(labels=srcWords[:opt.max_seq_length],
                                             unkWord=onmt.Constants.UNK_WORD)]
                tgt += [tgtDict.convertToIdx(labels=tgtWords[:opt.max_seq_length],
                                             unkWord=onmt.Constants.UNK_WORD,
                                             bosWord=onmt.Constants.BOS_WORD,
                                             eosWord=onmt.Constants.EOS_WORD)]
                sizes += [opt.max_seq_length]
                truncated += 1
            count += 1

            if count % opt.report_every == 0:
                print('... %d sentences read' % count)

    print('Kept %d sentences in %d (%d truncated due to length == 0 or > %d)' %
          (len(src), count, truncated, opt.max_seq_length))

    return src, tgt, sizes

def shuffle_data(src, tgt, sizes):
    print('... shuffling sentences')
    torch.manual_seed(opt.seed)
    perm = torch.randperm(len(src))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]
    sizes = [sizes[idx] for idx in perm]
    return src, tgt, sizes

def sort_by_length(src, tgt, sizes):
    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]
    sizes = [sizes[idx] for idx in perm]
    return src, tgt, sizes

def get_corpus_hist(sizes, name="train"):
    """
    get corpus histogram
    """
    import matplotlib.pyplot as plt
    import numpy as np
    n, bins, patches = plt.hist(sizes, 20, facecolor='green', alpha=0.5)
    plt.xlabel('Length')
    plt.ylabel('Freq')
    plt.xticks(np.arange(0, 150, 10), rotation=45)
    plt.title(r'Histogram')

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig('hist_%s.pdf' % name)
    plt.close()


def main():

    dicts = {}
    dicts['src'], dicts['tgt'] = initVocabulary(opt.train_file, opt.src_vocab, opt.src_vocab_size,
                                                opt.tgt_vocab, opt.tgt_vocab_size)

    print('Preparing training data...')
    all_data = {}
    all_data['src'], all_data['tgt'], all_data['sizes'] = makeData(opt.train_file,
                                                                   dicts['src'], dicts['tgt'])
    if opt.shuffle > 0:
        all_data['src'], all_data['tgt'], all_data['sizes'] = shuffle_data(all_data['src'],
                                                                           all_data['tgt'],
                                                                           all_data['sizes'])
    train = {}
    train['src'] = all_data['src'][:-591]
    train['tgt'] = all_data['tgt'][:-591]
    train_sizes = all_data['sizes'][:-591]
    train['src'], train['tgt'], train_sizes = sort_by_length(train['src'], train['tgt'], train_sizes)
    valid = {}
    valid['src'] = all_data['src'][-591:]
    valid['tgt'] = all_data['tgt'][-591:]
    valid_sizes = all_data['sizes'][-591:]
    valid['src'], valid['tgt'], valid_sizes = sort_by_length(valid['src'], valid['tgt'], valid_sizes)
    # get_corpus_hist(train_sizes)
    # get_corpus_hist(valid_sizes, name='valid')
    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')


    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': train,
                 'valid': valid}
    torch.save(save_data, opt.save_data + '.train.pt')


if __name__ == "__main__":
    main()
