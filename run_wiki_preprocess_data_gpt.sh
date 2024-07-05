python tools/preprocess_data.py \
       --input data/enwiki-latest-single-file/wiki_all.json \
       --output-prefix data/enwiki-latest-single-file/my-gpt2 \
       --vocab-file data/resources/gpt2-vocab.json \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file data/resources/gpt2-merges.txt \
       --workers 56 \
       --append-eod