import argparse
from text2phonemesequence import Text2PhonemeSequence

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", default="filelists/ljs_audio_text_test_filelist.txt")
  parser.add_argument("--output_file", default="filelists/ljs_audio_text_test_filelist_phoneme_sequence.txt")
  parser.add_argument("--language", default="eng-us")
  parser.add_argument("--cuda", default=False, action='store_true')
  parser.add_argument("--pretrained_g2p_model", default="charsiu/g2p_multilingual_byT5_tiny_16_layers_100")
  parser.add_argument("--tokenizer", default="google/byt5-small")
  parser.add_argument("--batch_size", default=64)

  args = parser.parse_args()

  # Load Text2PhonemeSequence
  model = Text2PhonemeSequence(pretrained_g2p_model=args.pretrained_g2p_model, tokenizer=args.tokenizer, language=args.language, is_cuda=args.cuda)
  
  # Processing data
  model.infer_dataset(input_file = args.input_file, output_file=args.output_file, batch_size=args.batch_size)
  





   