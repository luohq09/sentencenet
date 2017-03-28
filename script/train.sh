#!/usr/bin/env bash

# 获取模块工作目录的绝对路径
CWD="`pwd`/$( dirname "${BASH_SOURCE[0]}" )/../src"

cd "${CWD}" && python sentencenet_train.py \
    --pretrained_word_embedding_file="/Users/luohuaqing/sentence_data/data/embedding_sample.txt" \
    --train_data_file="/Users/luohuaqing/sentence_data/data/paraphrase/sample_data/a_train_seg.txt" \
    --dev_data_file="/Users/luohuaqing/sentence_data/data/paraphrase/sample_data/a_dev_seg.txt" \
    --out_dir="/Users/luohuaqing/sentence_data/data/paraphrase/sample_data/tf_model" \
    --evaluate_every=10 \
    --num_epochs=200
