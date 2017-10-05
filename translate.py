#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------

from __future__ import print_function
from __future__ import division
from __future__ import print_function
import onmt
import torch
import argparse
import math
import sys
import time

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src',
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions""")
parser.add_argument('-output_gold', default='gold.txt',
                    help="""Path to output the gold answers""")
parser.add_argument('-beam_size',  type=int, default=15,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=10,
                    help='Batch size')
parser.add_argument('-max_sent_length', default=140,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had the highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
# parser.add_argument('-phrase_table',
#                     help="""Path to source-target dictionary to replace UNK
#                     tokens. See README.md for the format of this file.""")
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")

parser.add_argument('-gpu', type=int, default=0,
                    help="Device to run on")

opt = parser.parse_args()

def replace_wide_chars(chars):
    # do not replace wide chars
    return chars
    wide_char_range = range(0xff01, 0xff5f)
    return_chars = list()
    for c in chars:
        if ord(c) in wide_char_range:
            c = unichr(ord(c) - 0xfee0)
        return_chars.append(c)
    return return_chars

def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal/wordsTotal)))

def decode_file(translator):
    outF = open(opt.output, 'w')
    # out_gold_file = open(opt.output_gold, 'w')
    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0., 0., 0., 0.
    srcBatch, tgtBatch = [], []
    count = 0
    start_time = time.time()
    for line in open(opt.src).readlines():
        line_split = line.decode('utf8').strip().split('\t')
        srcTokens = replace_wide_chars(list(line_split[0])[:opt.max_sent_length])
        srcBatch += [srcTokens]
        if len(line_split) > 1: # has gold tokens
            tgtTokens = list(line_split[1])
            tgtBatch += [tgtTokens]

        if len(srcBatch) < opt.batch_size:
            continue

        predBatch, predScore, goldScore = translator.translate(srcBatch, tgtBatch)

        predScoreTotal += sum(score[0] for score in predScore)
        predWordsTotal += sum(len(x[0]) for x in predBatch)
        if len(line_split) > 1: # has gold tokens
            goldScoreTotal += sum(goldScore)
            goldWordsTotal += sum(len(x) for x in tgtBatch)

        for b in range(len(predBatch)):
            count += 1
            outF.write("".join(predBatch[b][0]).encode('utf8')  + '\n')
            # out_gold_file.write("".join(tgtBatch[b]).encode('utf8') + '\n')
            if count % 1000 == 0:
                print("Decoded %d sents. Elapsed time %d s." % (count, time.time() - start_time))
            if opt.verbose:
                srcSent = ' '.join(srcBatch[b])
                if translator.tgt_dict.lower:
                    srcSent = srcSent.lower()
                print('SENT %d: %s' % (count, srcSent))
                print('PRED %d: %s' % (count, " ".join(predBatch[b][0])))
                print("PRED SCORE: %.4f" % predScore[b][0])

                if len(tgtBatch[b]) > 0:
                    tgtSent = ' '.join(tgtBatch[b])
                    if translator.tgt_dict.lower:
                        tgtSent = tgtSent.lower()
                    print('GOLD %d: %s ' % (count, tgtSent))
                    print("GOLD SCORE: %.4f" % goldScore[b])

                if opt.n_best > 1:
                    print('\nBEST HYP:')
                    for n in range(opt.n_best):
                        print("[%.4f] %s" % (predScore[b][n], " ".join(predBatch[b][n])))

                print('')

        srcBatch, tgtBatch = [], []

    reportScore('PRED', predScoreTotal, predWordsTotal)
    if goldWordsTotal > 0:
        reportScore('GOLD', goldScoreTotal, goldWordsTotal)


def decode_stream(translator):
    opt.batch_size = 1
    srcTokens = []
    org_input = list(raw_input("Input sentence:").decode('utf8'))
    clean_input = replace_wide_chars(org_input)
    srcTokens.append(clean_input)

    while len(srcTokens[0]) > 0:
        predBatch, _, _ = translator.translate(srcTokens, [])
        predicted_words = predBatch[0][0]
        print (''.join(predicted_words))

        srcTokens = []
        org_input = list(raw_input("Input sentence:").decode('utf8'))
        clean_input = replace_wide_chars(org_input)
        srcTokens.append(clean_input)

if __name__ == "__main__":
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    sys.stdout.write("Loading model file... ")
    sys.stdout.flush()
    translator = onmt.Translator(opt)
    print("Done.")
    if opt.src:
        decode_file(translator)
    else:
        print("Decode from prompt, input empty string to terminate. ")
        decode_stream(translator)
