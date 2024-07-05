python tools/preprocess_data.py \
       --input data/enwiki-latest-single-file/wiki_all.json \
       --output-prefix data/enwiki-latest-single-file/my-bert \
       --vocab-file data/resources/bert-large-uncased-vocab.txt \
       --tokenizer-type BertWordPieceLowerCase \
       --workers 56 \
       --split-sentences
