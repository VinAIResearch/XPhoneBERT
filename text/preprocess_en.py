import os
import random
import argparse
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from cleaners import english_cleaners
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="filelists/ljs_audio_text_test_filelist.txt")
    parser.add_argument("--output_file", default="filelists/ljs_audio_text_test_filelist_phoneme.txt")
    parser.add_argument("--language", default="eng-us")
    parser.add_argument("--is_cuda", default=True)
    args = parser.parse_args()

    f = open(args.input_file, 'r')
    list_lines = f.readlines()
    f.close()
    
    list_datas = []
    for line in tqdm(list_lines):
        line = line.strip()
        temp = line.split("|")
        assert len(temp) == 2
        file_name = temp[0].split('/')[-1]
        temp[1] = english_cleaners(temp[1].strip())
        list_words = word_tokenize(temp[1])
        list_datas.append("XPhoneBERT_EN/" + file_name + "|" + " ".join(list_words))

    f = open(args.output_file, 'w')
    for line in list_datas:
        f.write(line + "\n")
    f.close()



