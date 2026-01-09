import spacy
from collections import Counter

'''The Vocabulary Class is for tokenizing the text in an email 
for the data to be processed by the model
'''
class Vocabulary:
    '''
    @:param
    freq_threshold: Words that appear fewer times than this are ignored
    '''
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold

        #Define the mapping dictionaries
        # index_to_string: {0: "<PAD>", 1: "<SOS>", ...}
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

        # string_to_index: {"<PAD>": 0, "<SOS>": 1, ...}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

        # Load spacy english tokenizer
        self.spacy_eng = spacy.load("en_core_web_sm")

    def __len__(self):
        # Return length of vocabulary
        return len(self.itos)

    ''' Tokenize simple English text 
    @:param
    text: text to be tokenized
    '''
    def tokenizer_eng(self, text):
        # Return a list of strings (lower case)
        characters = []
        for token in self.spacy_eng.tokenizer(text):
            characters.append(token.text.lower())
        return characters

    ''' Build the vocab from a list of sentences 
    @:param
    sentence_list: list of sentences
    '''
    def build_vocabulary(self, sentence_list):
        # count frequencies of all words
        frequencies = Counter()
        idx = 4  # the start index for new word since 0-3 are taken

        for sentence in sentence_list:
            sentence = self.tokenizer_eng(sentence)
            frequencies.update(sentence)

        # add words to the dictionary if pass the threshold
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
            pass

    ''' Convert text to a list of integers 
    @:param
    text: text to be converted
    '''
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        # if word is in stoi, use that index. If not, use the index <UNK>
        result = []
        for word in tokenized_text:
            if word in self.stoi:
                result.append(self.stoi[word])
            else:
                result.append(self.stoi["<UNK>"])

        return result