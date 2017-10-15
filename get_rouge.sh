#!/bin/bash -x 
# REF=tgt_test.txt
# PRED=pred.txt
# calculate rouge score
# perl ROUGE_with_ranked.pl 1 N $REF $PRED
# perl ROUGE_with_ranked.pl 2 N $REF $PRED R
# perl ROUGE_with_ranked.pl 1 L $REF $PRED
python get_python_rouge.py -ans tgt_test.txt -pred pred.txt -source test.txt -output_ranked_r2 ranked_pred.txt
