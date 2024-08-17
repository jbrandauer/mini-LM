import re

def _word_to_index(vocab: list[str]):   
    word_to_index = {}
    for i, word in enumerate(vocab):
        word_to_index[word] = i
    return word_to_index

def _indices_lists(words: list[str],
                     vocab: list[str]):
    word_to_index = _word_to_index(vocab)
    indices_list = []
    for word in words:
        indices_list.append(word_to_index[word])
    return indices_list

def tokenize(text: str):
    words = re.split(r"\b", text)
    vocab = set(words)
    return _indices_lists(words, vocab), len(vocab)

if(__name__=="__main__"):
    with open("shakespeare.rtf", "r") as f:
        text = f.read()
    print("tokenized text: ", tokenize(text))
