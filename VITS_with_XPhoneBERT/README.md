## <a name="introduction"></a> VITS extended with our XPhoneBERT encoder

We provide a procedure to help you train the strong text-to-speech baseline [VITS](https://github.com/jaywalnut310/vits) with our XPhoneBERT encoder on [the LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/) or your own dataset.

### <a name="pre-require"></a> Pre-requisites

- Python >= 3.6
- Install python requirements. Please refer to [requirements.txt](requirements.txt): `
pip install -r requirements.txt`


### <a name="data-prepare"></a> Dataset preparation

#### For LJ Speech dataset

- Download and extract the LJ Speech dataset, then create a link to the dataset's wavs folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY`

- Convert LJ Speech text transcriptions into phoneme sequences using the following commands:
	- `python preprocess.py --input_file filelists/ljs_audio_text_train_filelist_preprocessed.txt --output_file filelists/ljs_audio_text_train_filelist_phoneme_sequence.txt --language eng-us --batch_size 64 --cuda`
	- `python preprocess.py --input_file filelists/ljs_audio_text_val_filelist_preprocessed.txt --output_file filelists/ljs_audio_text_val_filelist_phoneme_sequence.txt --language eng-us --batch_size 64 --cuda`


#### For another TTS dataset

- Prepare a dataset with the following structure:

```
Your_dataset_directory/
├── texts
    ├── training_texts_file.txt
    ├── validation_texts_file.txt
    ├── test_texts_file.txt
├── wavs
    ├── audio_1.wav
    ├── ...audio files in .wav format...
```

  - The `.txt` files in the `texts` directory contain text transcripts for training, validation and test, respectively. Each `.txt` file is formatted as:
  
    - `DUMMY/audio_1.wav|text transcript of the audio_1.wav , in which the text transcript is already word-segmented ( and text-normalized if applicable )`
    
    - For example, in the LJSpeech dataset: `DUMMY/LJ022-0023.wav|The overwhelming majority of people in this country know how to sift the wheat from the chaff in what they hear and what they read`

- Create a link to your dataset's `wavs` directory:

	- `ln -s /path/to/Your_dataset_directory/wavs DUMMY`

- Move/copy your `.txt` training, validation and test files to the `filelists` directory, and then run the `preprocess.py` file (similar to as run for the LJSpeech dataset), for example:

	- `cp /path/to/Your_dataset_directory/texts/*.txt /path/to/VITS_with_XPhoneBERT/filelists/`
	- `python preprocess.py --input_file filelists/training_texts_file.txt --output_file filelists/training_phoneme_sequences.txt --language <ISO-639-3-language-code> --batch_size 64 --cuda`
	- `python preprocess.py --input_file filelists/validation_texts_file.txt --output_file filelists/validation_phoneme_sequences.txt --language <ISO-639-3-language-code> --batch_size 64 --cuda`

`<ISO-639-3-language-code>` is the corresponding ISO 639-3 code of your own dataset's language. The ISO 639-3 codes of supported languages are available at [HERE](https://github.com/VinAIResearch/XPhoneBERT/blob/main/LanguageISO639-3Codes.md).

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

In our settings, we use 4 A100 GPUs (40GB each) for training. If users have a smaller computational resource, you need to decrease the batch size to avoid the Out-of-memory problem (it is advisable to decrease the initial learning rate accordingly). 

### <a name="infer"></a> Inference Example
See [inference.py](inference.py) file and justify corresponding paths in [inference.py](inference.py).

