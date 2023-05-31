import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence, cleaned_text_to_sequence
from transformers import AutoTokenizer, T5ForConditionalGeneration
from pyvi import ViTokenizer
from scipy.io.wavfile import write
import soundfile as sf
from segments import Tokenizer
from tqdm import tqdm
from text.cleaners import english_cleaners, expand_abbreviations
import nltk 

t = Tokenizer()
device = torch.device("cuda:{}".format(0))
model = T5ForConditionalGeneration.from_pretrained("/lustre/scratch/client/vinai/users/linhnt140/phonemeBERT/g2p_model/models--charsiu--g2p_multilingual_byT5_small_100/snapshots/ed5cf99707fb39f5f3ccc461530d56b6fb531fd3")
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained('/lustre/scratch/client/vinai/users/linhnt140/phonemeBERT/g2p_model/models--google--byt5-small/snapshots/ce8f3a48ed7676af36476a01fb01f95ea529599c')
punctuation = list('.?!,:;-()[]{}<>"') + list("'/‘”“/&#~@^|") + ['...', '*']

def phoneme_segmentation(text):
    try:
        text = t(text, ipa=True)
    except:
        text = t(text)
    text = text.replace('#', "▁")
    return text

def get_text(text, model, tokenizer, tokenizer_mphonebert, language, hps):
    # text_norm = text_to_sequence(text, hps.data.text_cleaners)
    text = english_cleaners(text)
    raw_words = nltk.word_tokenize(text)
    # raw_words = text.split(' ')
    print(raw_words)
    words = [language + ": " + i.lower().replace('_', ' ') for i in raw_words]
    out = tokenizer(words,padding=True,add_special_tokens=False,return_tensors='pt').to(device)
    preds = model.generate(**out,num_beams=1,max_length=50)
    phones = tokenizer.batch_decode(preds.tolist(),skip_special_tokens=True) 
    assert len(phones) == len(raw_words)
    for i in range(len(phones)):
        if raw_words[i] in punctuation:
            phones[i] = raw_words[i]
    print(phones)
    text_phones = ' '.join(phones)
    text_norm = phoneme_segmentation(text_phones)
    tokenized_text = tokenizer_mphonebert(text_norm)
    input_ids = tokenized_text['input_ids']
    attention_mask = tokenized_text['attention_mask']
    input_ids = torch.LongTensor(input_ids).to(device)
    attention_mask = torch.LongTensor(attention_mask).to(device)
    # if hps.data.add_blank:
    #     text_norm = commons.intersperse(text_norm, 0)
    # text_norm = torch.LongTensor(text_norm)
    return input_ids, attention_mask

hps = utils.get_hparams_from_file("./configs/lj_base_xphonebert.json")
tokenizer_mphonebert = AutoTokenizer.from_pretrained(hps.bert)
net_g = SynthesizerTrn(
    hps.bert,
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("/lustre/scratch/client/vinai/users/linhnt140/phonemeBERT/vits_bert/logs/lj_base_xphonebert/G_161200.pth", net_g, None)

# text = 'Tottenham tiếp đón Chelsea trên sân nhà ở vòng 25, Ngoại hạng Anh'
f = open('/lustre/scratch/client/vinai/users/linhnt140/phonemeBERT/vits_en/filelists/ljs_audio_text_test_filelist.txt', 'r')
list_lines = f.readlines()
f.close()
for line in tqdm(list_lines):
    line = line.strip().split('|')
    assert len(line) == 2

    stn_tst, attention_mask = get_text(line[-1], model, tokenizer, tokenizer_mphonebert, '<eng-us>', hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)

        # x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        attention_mask = attention_mask.cuda().unsqueeze(0)
        audio = net_g.infer(x_tst, attention_mask, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

    sf.write('./out_lj_base_xphonebert/' + line[0].split('/')[-1], audio, hps.data.sampling_rate)
    ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))


