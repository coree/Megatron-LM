python tools/preprocess_data.py \
       --input data/enwiki-latest-single-file/wiki_all.json \
       --output-prefix data/enwiki-latest-single-file/bert-uncased \
       --vocab-file data/models/megatron_bert_345m_v0.1_uncased/bert-large-uncased-vocab.txt \
       --tokenizer-type BertWordPieceLowerCase \
       --workers 56 \
       --split-sentences
