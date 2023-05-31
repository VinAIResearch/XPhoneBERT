# XPhoneBERT :  A Pre-trained Multilingual Model for Phoneme Representations for Text-to-Speech

### Linh The Nguyen, Thinh Pham, and Dat Quoc Nguyen

- [Introduction](#introduction)
- [Pre-requisites](#pre-require)
- [Training example](#training)
- [Inference](#infer)

# <a name="introduction"></a> Introduction
In our [paper](***), we present XPhoneBERT, a first pre-trained multilingual model for phoneme representations for text-to-speech(TTS).

Our XPhoneBERT has the same model architecture as BERT-base, trained using the RoBERTa pre-training approach on 330M phoneme-level sentences from nearly 100 languages and locales. Experimental results show that employing XPhoneBERT as an input phoneme encoder significantly boosts the performance of a strong neural TTS model in terms of naturalness and prosody and also helps produce fairly high-quality speech with limited training data.

# <a name="pre-require"></a> Pre-requisites
Almost all steps are similar with the original VITS but there are some small changes.

0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
0. Preprare dataset 
    1. Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
    1. If you're using your own dataset and language, you need to convert your files into the same format as the LJ Speech dataset. Ensure that your dataset has undergone word segmentation, and text normalization before renaming the dataset folder or create a link to it using the following command: `ln -s /path/to/your_datasets/wavs XPhoneBERT_EN`.
    1. Next, convert the datasets into phoneme sequences using the following command: `python preprocess.py --input_file path/to/input_file --output_file path/to/output_file --language language_code --cuda`.
0. Build Monotonic Alignment Search.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace

```


# <a name="training"></a> Training Example
```sh
# LJ Speech
python train.py -c configs/ljs_base_xphonebert.json -m ljs_base_xphonebert

# Your own dataset: You need to adjust the config file to appropriate with your dataset.
```

# <a name="infer"></a> Inference Example
See [inference.py] (inference.py) file

For users who want to use our XphoneBERT for other models or purposes, we provide a library [text2phonemesequence](https://github.com/thelinhbkhn2014/Text2PhonemeSequence). This library helps to convert raw text into phoneme sequences that can be used as input for our XPhoneBERT.
