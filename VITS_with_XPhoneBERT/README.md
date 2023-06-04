# VITS with XPhoneBERT

- [Introduction](#introduction)
- [Pre-requisites](#pre-require)
- [Training example](#training)
- [Inference](#infer)

# <a name="introduction"></a> Introduction
We provide a procedure to help you train [VITS](https://github.com/jaywalnut310/vits) with our XPhoneBERT on the LJSpeech dataset or your own dataset. Almost all steps are similar to the original [VITS](https://github.com/jaywalnut310/vits) but there are some small changes. 
# <a name="pre-require"></a> Pre-requisites

0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
0. Preprare dataset 
    1. Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs XPhoneBERT_EN`
    1. If you're using your own dataset and language, you need to convert your files into the same format as the LJ Speech dataset. Ensure that your dataset has undergone word segmentation, and text normalization before renaming the dataset folder or create a link to it using the following command: `ln -s /path/to/your_datasets/wavs DUMMY`.
    1. Next, convert the datasets into phoneme sequences using the following command: `python preprocess.py --input_file path/to/input_file --output_file path/to/output_file --language language_code --batch_size 64 --cuda`.
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
See [inference.py](inference.py) file

[//]: # (For users who want to use our XphoneBERT for other models or purposes, we provide a library [text2phonemesequence]&#40;https://github.com/thelinhbkhn2014/Text2PhonemeSequence&#41;. This library helps to convert raw text into phoneme sequences that can be used as input for our XPhoneBERT.)
