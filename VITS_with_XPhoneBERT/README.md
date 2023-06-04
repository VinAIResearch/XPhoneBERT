## <a name="introduction"></a> VITS extended with our XPhoneBERT encoder

We provide a procedure to help you train the strong text-to-speech baseline [VITS](https://github.com/jaywalnut310/vits) with our XPhoneBERT encoder on [the LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/) or your own dataset.

### <a name="pre-require"></a> Pre-requisites

- Python >= 3.6
- Install python requirements. Please refer to [requirements.txt](requirements.txt): `
pip install -r requirements.txt`


### <a name="data-prepare"></a> Dataset preparation

#### For LJ Speech dataset

- Download and extract the LJ Speech dataset, then create a link to the dataset's wavs folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY`

- Convert LJ Speech text transcriptions into phoneme sequences using following commands:
	- `python preprocess.py --input_file filelists/ljs_audio_text_train_filelist_preprocessed.txt --output_file filelists/ljs_audio_text_train_filelist_phoneme_sequence.txt --language eng-us --batch_size 64 --cuda`
	- `python preprocess.py --input_file filelists/ljs_audio_text_val_filelist_preprocessed.txt --output_file filelists/ljs_audio_text_val_filelist_phoneme_sequence.txt --language eng-us --batch_size 64 --cuda`


#### For another TTS dataset

- Prepare a dataset with the following structure:
  - Dataset_folder:
    - text
      - training_texts_file.txt
      - validation_texts_file.txt
      - test_texts_file.txt
    - wavs
      - audio files in .wav format
  - The `.txt` files in the `text` directory contain text transcripts for training, validation, and test respectively. The format for each `.txt` file is:
    - `DUMMY/file_a.wav|text transcript of the file_a.wav (that is already word-segmented and text-normalized if applicable)`
  - For example, in the LJSpeech dataset: `DUMMY/LJ022-0023.wav|The overwhelming majority of people in this country know how to sift the wheat from the chaff in what they hear and what they read`
- Move `.txt` files to the `filelists` directory, then create a link to the dataset's wavs folder and run the `preprocess.py` file similar to the LJspeech dataset.


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

