""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols
import re
from segments import Tokenizer


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
_whitespace_re = re.compile(r'\s+')
t = Tokenizer()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  clean_text = _clean_text(text, cleaner_names)
  for symbol in clean_text:
    symbol_id = _symbol_to_id[symbol]
    sequence += [symbol_id]
  return sequence


def cleaned_text_to_sequence(cleaned_text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  list_phones = []
  list_words = cleaned_text.split(' ')
  for word in list_words:
    try:
      list_p = t(word, ipa=True).split(' ')
    except:
      list_p = t(word).split(' ')
    list_phones = list_phones + list_p
    list_phones = list_phones + [' ']
  list_phones = list_phones[:-1]
  sequence = [_symbol_to_id[symbol] for symbol in list_phones]
  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


if __name__ == '__main__':
    cleaned_text_to_sequence("˨˨@w @tɕ@iː˨˩ˀ @ɓ@e˨˩ˀ@ɲ @v@iə˨˨@m @ɣ@aː˨˨@n")