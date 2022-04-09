import torch
from torchtext import data
from torchtext.vocab import Vectors
import spacy
import pandas as pd
import re
from sklearn.datasets import fetch_20newsgroups
import random
from collections import Counter
import functools
import numpy as np
from sklearn.metrics import accuracy_score

PAD_TOKEN = '<PAD>'
WHITESPACE_REGEXP = re.compile(r'\s+')

class Dataset(object):
    def __init__(self, args, embedding, tokenizer, entity_linker):
        self.args = args
        self.embedding = embedding
        self.word_tokenizer = tokenizer
        self.entity_linker = entity_linker

        self.train_data = None
        self.test_data = None

        self.fields = None

        self.train_df = None
        self.train_examples = None
        self.val_examples = None
        self.test_examples = None
        self.dataset_train = None
        self.dataset_test = None
        self.dataset_val = None
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None

        self.val_file = None

        self.vocab = []
        self.word_embeddings = {}

    @functools.lru_cache(maxsize=None)
    def tokenize(text):
        return self.tokenizer.tokenize(text)

    @functools.lru_cache(maxsize=None)
    def detect_mentions(text):
        return self.entity_linker.detect_mentions(text)

    def create_numpy_sequence(source_sequence, length, dtype):
        ret = np.zeros(length, dtype=dtype)
        source_sequence = source_sequence[:length]
        ret[:len(source_sequence)] = source_sequence
        return ret

    def parse_label(self, label):
        """
        Get the actual labels from label string
        Input: label (string) : labels of the form '__label__2'
        Returns: label (int) : integer value corresponding to label string
        """
        # return int(label.strip()[-1])
        label = ''.join(label.strip())
        return int(label.split('_')[-1])

    def get_pandas_df(self, filename):
        """
        Load the data_fed into Pandas.DataFrame object
        This will be used to convert data_fed to torchtext object
        """
        with open(filename, 'r') as datafile:     
            data = [line.strip().split(',', maxsplit=1) for line in datafile]
            data_text = list(map(lambda x: x[1], data))
            data_label = list(map(lambda x: self.parse_label(x[0]), data))

        full_df = pd.DataFrame({"text": data_text, "label": data_label})
        return full_df
    
    def load_20newsgroup_data(self, dev_size, w2v_file, train_file, test_file=None, val_file=None):
        if self.args.dataset != '20ng':
            exit()
        # self.train_data
        # self.test_data
        # 为包含 '(text, label)' 的内容列表, 'text' 是内容，'label' 是标签
        self.train_data = []
        self.test_data = []
        for fold in ('train', 'test'):
            dataset = fetch_20newsgroups(subset=self.args.dataset, shuffle=False)
            for text, label in zip(dataset['data_fed'], dataset['target']):
                text = self.normalize_text(text)
                if fold == 'train':
                    self.train_data.append((text, label))
                else:
                    self.test_data.append((text, label))

        dev_size = int(len(self.train_data) * dev_size)
        random.shuffle(self.train_data)

        self.train_text = [text for text, _ in self.train_data]
        self.train_label = [label for _, label in self.train_data]

        # word_counter, entity_counter
        word_counter = Counter()
        entity_counter = Counter()
        word_counter.update(t.text for t in self.word_tokenizer.tokenize(self.train_text))
        entity_counter.update(m.title for m in self.entity_linker.detect_mentions(self.train_text))

        # word_set, entity_vocab, self.word_vocab, self.entity_vocab
        word_set = [word for word, count in word_counter.items() if count >= self.args.min_count]
        self.word_vocab = {word: index for index, word in enumerate(word_set, 1)}
        self.word_vocab[PAD_TOKEN] = 0
        entity_titles = [title for title, count in entity_counter.items() if count >= self.args.min_count]
        self.entity_vocab = {title: index for index, title in enumerate(entity_titles, 1)}
        self.entity_vocab[PAD_TOKEN] = 0


        ############################################################
        self.train_text = [text for text, _ in self.train_data if text in self.word_vocab]
        self.train_label = [label for text, label in self.train_data if text in self.word_vocab]

        word_ids = [self.word_vocab[token.text]
                    for token in self.word_tokenizer.tokenize(self.train_text)
                    if token.text in self.word_vocab]
        self.entity_ids = []
        self.prior_probs = []
        for mention in self.entity_linker.detect_mentions(self.train_text):
            if mention.title in self.entity_vocab:
                self.entity_ids.append(self.entity_vocab[mention.title])
                self.prior_probs.append(mention.prior_prob)


        # self.entity_ids = self.create_numpy_sequence(entity_ids, self.args.max_entity_length, np.int)
        # self.prior_probs = self.create_numpy_sequence(prior_probs, self.args.max_entity_length, np.float32)
        #####################################################################

        # Creating Field for data_fed
        TEXT = data.Field(sequential=True, tokenize=None, lower=True, fix_length=self.args.max_sen_len)
        LABEL = data.Field(sequential=False, use_vocab=False)
        self.fields = [("text", TEXT), ("label", LABEL)]
        # Load data_fed from pd.DataFrame into torchtext.data_fed.Dataset
        self.train_df = self.get_pandas_df(train_file)
        self.train_examples = [data.Example.fromlist(i, self.fields) for i in self.train_df.values.tolist()]
        self.dataset_train = data.Dataset(self.train_examples, self.fields)

        test_df = self.get_pandas_df(test_file)
        self.test_examples = [data.Example.fromlist(i, self.fields) for i in test_df.values.tolist()]
        self.dataset_test = data.Dataset(self.test_examples, self.fields)
        # If validation file exists, load it. Otherwise get validation data_fed from training data_fed
        self.val_file = val_file
        if self.val_file:
            val_df = self.get_pandas_df(val_file)
            self.val_examples = [data.Example.fromlist(i, self.fields) for i in val_df.values.tolist()]
            self.dataset_val = data.Dataset(self.val_examples, self.fields)
        else:
            self.dataset_train, self.dataset_val = self.dataset_train.split(split_ratio=0.8)
        if w2v_file:
            TEXT.build_vocab(self.dataset_train, vectors=Vectors(w2v_file))
        self.word_embeddings = TEXT.vocab.vectors
        self.vocab = TEXT.vocab

        self.train_iterator = data.BucketIterator(
            (self.dataset_train),
            batch_size=self.args.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)

        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (self.dataset_val, self.dataset_test),
            batch_size=self.args.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)

        print("Loaded {} training examples".format(len(self.dataset_train)))
        print("Loaded {} test examples".format(len(self.dataset_test)))
        print("Loaded {} validation examples".format(len(self.dataset_val)))
        print("数据集加载成功！")


    def normalize_text(text):
        text = text.lower()
        text = re.sub(WHITESPACE_REGEXP, ' ', text)

        # remove accents: https://stackoverflow.com/a/518232
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        text = unicodedata.normalize('NFC', text)

        return text






def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
