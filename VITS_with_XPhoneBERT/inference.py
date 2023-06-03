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
from transformers import AutoTokenizer, T5ForConditionalGeneration
from scipy.io.wavfile import write
import soundfile as sf
from text2phonemesequence import Text2PhonemeSequence
from tqdm import tqdm


def get_inputs(text, model, tokenizer_xphonebert):
    phones = model.infer_sentence(text)
    tokenized_text = tokenizer_xphonebert(phones)
    input_ids = tokenized_text['input_ids']
    attention_mask = tokenized_text['attention_mask']
    input_ids = torch.LongTensor(input_ids).cuda()
    attention_mask = torch.LongTensor(attention_mask).cuda()
    return input_ids, attention_mask

hps = utils.get_hparams_from_file("./configs/lj_base_xphonebert.json")
tokenizer_xphonebert = AutoTokenizer.from_pretrained(hps.bert)
# Load Text2PhonemeSequence
model = Text2PhonemeSequence(pretrained_g2p_model='charsiu/g2p_multilingual_byT5_tiny_16_layers_100', language='vie-n', is_cuda=True)
net_g = SynthesizerTrn(
    hps.bert,
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("./logs/lj_base_xphonebert/G_161200.pth", net_g, None)

f = open('./filelists/ljs_audio_text_test_filelist_phones.txt', 'r')
list_lines = f.readlines()
f.close()
for line in tqdm(list_lines):
    line = line.strip().split('|')
    assert len(line) == 2

    stn_tst, attention_mask = get_text(line[-1], model, tokenizer_xphonebert)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        attention_mask = attention_mask.cuda().unsqueeze(0)
        audio = net_g.infer(x_tst, attention_mask, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

    sf.write('./out_ljspeech/' + line[0], audio, hps.data.sampling_rate)
    ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
    


