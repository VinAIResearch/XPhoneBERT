import os
import py_vncorenlp
import argparse
from vinorm import TTSnorm
import random

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="filelists/train_vn.txt")
    parser.add_argument("--output_file", default="filelists/train_vn_phoneme.txt")
    parser.add_argument("--language", default="vie-n")
    parser.add_argument("--is_cuda", default=False)
    args = parser.parse_args()

    word_tokenize = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/home/vinai/Documents/VnCoreNLP_Wrapper')
    
    f = open(args.input_file, 'r')
    list_lines = f.readlines()
    f.close()
    
    list_datas = []
    for line in list_lines:
        line = line.strip()
        temp = line.split("|")
        assert len(temp) == 2
        file_name = temp[0].split("/")[-1]

        temp[1] = temp[1].strip()
        if temp[1][-1] == '.':
            temp[1] = temp[1][:-1]
        if has_numbers(temp[1]):
            output = word_tokenize.word_segment(TTSnorm(temp[1]))
        else:
            output = word_tokenize.word_segment(temp[1])

        list_datas.append("XPhoneBERT_vi/" + file_name + "|" + ' '.join(output))

    f = open(args.output_file, 'w')
    
    for line in list_datas:
        f.write(line + "\n")
    f.close()
    

