
#### Table of contents
1. [Introduction](#introduction)
2. [Using XPhoneBERT with `transformers`](#transformers)
	- [Installation](#install2)
	- [Example usage](#usage2)


# <a name="introduction"></a> XPhoneBERT :  A Pre-trained Multilingual Model for Phoneme Representations for Text-to-Speech 

In our [paper](https://arxiv.org/abs/2305.19709), we present XPhoneBERT, a first pre-trained multilingual model for phoneme representations for text-to-speech(TTS).

Our XPhoneBERT has the same model architecture as BERT-base, trained using the RoBERTa pre-training approach on 330M phoneme-level sentences from nearly 100 languages and locales. Experimental results show that employing XPhoneBERT as an input phoneme encoder significantly boosts the performance of a strong neural TTS model in terms of naturalness and prosody and also helps produce fairly high-quality speech with limited training data.

The general architecture and experimental results of XPhoneBERT can be found in our [paper](https://arxiv.org/abs/2305.19709):

    @inproceedings{xphonebert,
    title     = {{XPhoneBERT : A Pre-trained Multilingual Model for Phoneme Representations for Text-to-Speech},
    author    = {Linh The Nguyen, Thinh Pham, and Dat Quoc Nguyen},
    booktitle = {Proceedings of the 24th Annual Conference of the International Speech Communication Association (INTERSPEECH)},
    year      = {2023}
    }

**Please CITE** our paper when XPhoneBERT is used to help produce published results or is incorporated into other software.

## <a name="transformers"></a> Using XPhoneBERT with `transformers` 

### Installation <a name="install2"></a>
- Install `transformers` with pip: `pip install transformers`, or install `transformers` [from source](https://huggingface.co/docs/transformers/installation#installing-from-source).  <br /> 

- Before using XPhoneBERT, users need to convert text to phoneme sequences. To be convenient for users, we build a [Text2PhonemeSequence](https://github.com/thelinhbkhn2014/Text2PhonemeSequence) library that can be installed with pip: `pip install text2phonemesequence`.
- Note that sentences need to be performed word segmentation, and text normalization before using the `Text2PhonemeSequence` library.
### Example usage <a name="usage2"></a>

```python
from transformers import AutoModel, AutoTokenizer
from text2phonemesequence import Text2PhonemeSequence

tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")
xphonebert = AutoModel.from_pretrained("vinai/xphonebert-base")
# Load Text2PhonemeSequence
text2phone_model = Text2PhonemeSequence(pretrained_g2p_model='charsiu/g2p_multilingual_byT5_tiny_16_layers_100', language='eng-us', is_cuda=False)

# INPUT TEXT MUST BE ALREADY WORD-SEGMENTED AND TEXT NORMALIZED
sentence = 'it has used other treasury law enforcement agents on special experiments in building and route surveys in places to which the president frequently travels .'  
input_phonemes = text2phone_model.infer_sentence(sentence)

input_ids = tokenizer(input_phonemes, return_tensors="pt")

with torch.no_grad():
    features = xphonebert(**input_ids)
```

## License
    
	MIT License

	Copyright (c) 2023 VinAI Research

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.
