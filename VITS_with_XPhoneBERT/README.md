## <a name="introduction"></a> VITS extended with our XPhoneBERT encoder

We provide a procedure to help you train the strong text-to-speech baseline [VITS](https://github.com/jaywalnut310/vits) with our XPhoneBERT encoder on [the LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/) or your own dataset.

### <a name="pre-require"></a> Pre-requisites

- Python >= 3.6
- Install python requirements. Please refer to [requirements.txt](requirements.txt): `
pip install -r requirements.txt`


### Dataset preparation

#### For LJ Speech dataset

- Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY`

- Convert LJ Speech text transcriptions into phoneme sequences using following commands:
	- `python preprocess.py --input_file filelists/ljs_audio_text_train_filelist_preprocessed.txt --output_file filelists/ljs_audio_text_train_filelist_phoneme_sequence.txt --language eng-us --batch_size 64 --cuda`
	- `python preprocess.py --input_file filelists/ljs_audio_text_val_filelist_preprocessed.txt --output_file filelists/ljs_audio_text_val_filelist_phoneme_sequence.txt --language eng-us --batch_size 64 --cuda`


#### For another TTS dataset


.....


### Building Monotonic Alignment Search
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace
```


### <a name="training"></a> Training Example
```sh
# LJ Speech
python train.py -c configs/ljs_base_xphonebert.json -m ljs_base_xphonebert

# Your own dataset: You need to adjust the config file to appropriate with your dataset.
```

### <a name="infer"></a> Inference Example
See [inference.py](inference.py) file

