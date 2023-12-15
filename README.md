# <a name="introduction"></a> XPhoneBERT :  A Pre-trained Multilingual Model for Phoneme Representations for Text-to-Speech 

XPhoneBERT is the first pre-trained multilingual model for phoneme representations for text-to-speech(TTS). XPhoneBERT has the same model architecture as BERT-base, trained using the RoBERTa pre-training approach on 330M phoneme-level sentences from nearly 100 languages and locales. Experimental results show that employing XPhoneBERT as an input phoneme encoder significantly boosts the performance of a strong neural TTS model in terms of naturalness and prosody and also helps produce fairly high-quality speech with limited training data.

The general architecture and experimental results of XPhoneBERT can be found in [our INTERSPEECH 2023 paper](https://arxiv.org/abs/2305.19709):

    @inproceedings{xphonebert,
    title     = {{XPhoneBERT: A Pre-trained Multilingual Model for Phoneme Representations for Text-to-Speech}},
    author    = {Linh The Nguyen and Thinh Pham and Dat Quoc Nguyen},
    booktitle = {Proceedings of the 24th Annual Conference of the International Speech Communication Association (INTERSPEECH)},
    year      = {2023}
    }

**Please CITE** our paper when XPhoneBERT is used to help produce published results or is incorporated into other software.

## <a name="transformers"></a> Using XPhoneBERT with `transformers` 

### Installation <a name="install2"></a>

- Install `transformers` with pip: `pip install transformers`, or install `transformers` [from source](https://huggingface.co/docs/transformers/installation#installing-from-source). 

- Install `text2phonemesequence`: `pip install text2phonemesequence` <br>  Our [`text2phonemesequence`](https://github.com/thelinhbkhn2014/Text2PhonemeSequence) package is to convert text sequences into phoneme-level sequences, employed to construct our multilingual phoneme-level pre-training data. We build `text2phonemesequence` by incorporating the [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P/tree/main) and the [segments](https://pypi.org/project/segments/) toolkits that perform text-to-phoneme conversion and phoneme segmentation, respectively. 

- **Notes**

	-	Initializing `text2phonemesequence` for each language requires its corresponding ISO 639-3 code. The ISO 639-3 codes of supported languages are available at [HERE](https://github.com/VinAIResearch/XPhoneBERT/blob/main/LanguageISO639-3Codes.md).
	
	- `text2phonemesequence` takes a word-segmented sequence as input. And users might also perform text normalization on the word-segmented sequence before feeding into `text2phonemesequence`. When creating our pre-training data, we perform word and sentence segmentation on all text documents in each language by using the [spaCy](https://spacy.io) toolkit, except for Vietnamese where we employ the [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) toolkit. We also use the text normalization component from the [NVIDIA NeMo toolkit](https://github.com/NVIDIA/NeMo) for English, German, Spanish and Chinese, and the [Vinorm](https://github.com/v-nhandt21/Vinorm) text normalization package for Vietnamese.


### <a name="models2"></a> Pre-trained model 

Model | #params | Arch. | Max length | Pre-training data
---|---|---|---|---
[`vinai/xphonebert-base`](https://huggingface.co/vinai/xphonebert-base) | 88M | base | 512 | 330M phoneme-level sentences from nearly 100 languages and locales


### Example usage <a name="usage2"></a>

```python
from transformers import AutoModel, AutoTokenizer
from text2phonemesequence import Text2PhonemeSequence

# Load XPhoneBERT model and its tokenizer
xphonebert = AutoModel.from_pretrained("vinai/xphonebert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")

# Load Text2PhonemeSequence
# text2phone_model = Text2PhonemeSequence(language='eng-us', is_cuda=True)
text2phone_model = Text2PhonemeSequence(language='jpn', is_cuda=True)

# Input sequence that is already WORD-SEGMENTED (and text-normalized if applicable)
# sentence = "That is , it is a testing text ."  
sentence = "これ は 、 テスト テキスト です ."

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
