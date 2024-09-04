import nltk
from gensim.models import KeyedVectors
import random
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import re
import unicodedata
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Download necessary NLTK data files
# nltk.download('punkt')


class TextProcessor:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
# you can also try stemming and lemmatizers to improve performance
# https://towardsdatascience.com/text-preprocessing-with-nltk-9de5de891658

    def clean_text(self, text):
        text = unicodedata.normalize("NFD", text)
        text = text.lower()
        text = re.sub(r"[^0-9a-zA-Z?.,!:;]+", r" ", text)
        text = re.sub(r"(.)\1{3,}", r"\1", text)
        text = text.strip()
        return text

    def preprocess_text(self):
        with open(self.file_path, 'r') as f:
            corpus = f.read()
        sentences = sent_tokenize(self.clean_text(corpus))
        sentences = [sent for sent in sentences if not sent.lower(
        ).startswith('chapter') and not sent[0].isdigit()]
        return sentences

    def generate_ngrams(self, sentences, n):
        ngrams = []
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            tokens = ['<S>'] + tokens + ['</S>']
            # <s> helps in identifying the start of a sentence, allowing the model to learn the patterns and probabilities associated with words that commonly appear at the beginning of sentences
            # <\s> helps the model understand when the a sentence is complete and learn word sentences that commonly end sentences.
            sentence_ngrams = zip(*[tokens[i:] for i in range(n)])
            ngrams.extend([' '.join(ngram) for ngram in sentence_ngrams])
        return ngrams


class Vocabulary:
    def __init__(self, glove_path):
        self.glove_model = KeyedVectors.load_word2vec_format(
            glove_path, binary=False, no_header=True)
        self.word2idx = {}
        self.idx2word = []
        self.special_tokens = ['<UNK>', '<S>', '</S>']
        self._add_special_tokens()

    def _add_special_tokens(self):
        for token in self.special_tokens:
            self.add_word(token)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)

    def build_vocab(self, sentences):
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            for token in tokens:
                self.add_word(token)

    def get_glove_embeddings(self):
        embedding_dim = self.glove_model.vector_size
        embeddings = np.zeros((len(self.idx2word), embedding_dim))
        # embeddings in the shape of (vocab_size, embedding_dim)
        for idx, word in enumerate(self.idx2word):
            if word in self.glove_model:
                embeddings[idx] = self.glove_model[word]
            else:
                embeddings[idx] = np.random.normal(
                    scale=0.6, size=(embedding_dim,))
        return torch.tensor(embeddings, dtype=torch.float32)

    def index2word(self, idx):
        return self.idx2word[idx]

    def word2index(self, word):
        return self.word2idx.get(word, self.word2idx['<UNK>'])

    def __len__(self):
        return len(self.word2idx)


class TextDataset(Dataset):
    def __init__(self, ngrams, vocab):
        self.ngrams = ngrams
        self.vocab = vocab

    def __len__(self):
        return len(self.ngrams)

    def __getitem__(self, idx):
        ngram = self.ngrams[idx].split()
        context_idxs = [self.vocab.word2index(word) for word in ngram[:-1]] # [:-1] gets the context words
        target_idx = self.vocab.word2index(ngram[-1]) # [-1] gets the target word
        return torch.tensor(context_idxs, dtype=torch.long), torch.tensor(target_idx, dtype=torch.long)

# Read and preprocess the corpus
"""
file_path = 'Auguste_Maquet.txt'
glove_path = 'glove.6B.50d.txt'

processor = TextProcessor(file_path)
sentences = processor.preprocess_text()

random.seed(25)
random.shuffle(sentences)

val_len = 10000
test_len = 20000

train_sents = sentences[val_len + test_len:]
validation_sents = sentences[:val_len]
test_sents = sentences[val_len:val_len + test_len]

train_ngrams = processor.generate_ngrams(train_sents, 5 + 1)
validation_ngrams = processor.generate_ngrams(validation_sents, 5 + 1)
test_ngrams = processor.generate_ngrams(test_sents, 5 + 1)

# print(train_ngrams[:5])

vocab = Vocabulary(glove_path)
vocab.build_vocab(train_sents)
embeddings = vocab.get_glove_embeddings()

# Create datasets and dataloaders
train_dataset = TextDataset(train_ngrams, vocab)
validation_dataset = TextDataset(validation_ngrams, vocab)
test_dataset = TextDataset(test_ngrams, vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
"""