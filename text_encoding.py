import re

def _word_to_index(vocab: list[str]):   
    word_to_index = {}
    index_to_word = {}
    for i, word in enumerate(vocab):
        word_to_index[word] = i
        index_to_word[i] = word
    return word_to_index, index_to_word

def decode_sequence(indices: list[int],
                    index_to_word: dict[int, str])->str:
    output_list = list()
    for id in indices:
        output_list.append(index_to_word[id])
    return output_list

def encode_sequence(input_list: list[str],
                    word_to_index: dict[str, int])->list[int]:
    output_list = list()
    for word in input_list: 
        output_list.append(word_to_index[word])
    return output_list

def _indices_lists(words: list[str],
                     vocab: list[str]):
    word_to_index, index_to_word = _word_to_index(vocab)
    indices_list = []
    for word in words:
        indices_list.append(word_to_index[word])
    return indices_list, word_to_index, index_to_word

def tokenize(text: str):
    words = re.split(r"\b", text)
    vocab = set(words)
    return _indices_lists(words, vocab), len(vocab)

if(__name__=="__main__"):
    with open("shakespeare.rtf", "r") as f:
        text = f.read()
    print("tokenized text: ", tokenize(text))
