import torch
from torchtext import data
from torchtext.vocab import Vectors
import spacy
import pandas as pd
import re
from sklearn.metrics import accuracy_score

import functools
import logging
import os
import random
import re
import unicodedata
from collections import Counter
import numpy as np
from bs4 import BeautifulSoup
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
import sys
sys.path.append('/home/user/Documents/mmm/paper/wikipedia2vec-master/examples/fed_avg/utils_fed')
import options
sys.path.append('/home/user/Documents/mmm/paper/wikipedia2vec-master/examples/fed_avg/data_fed')
from reformat_20news_bydate import *

PAD_TOKEN = '<PAD>'
WHITESPACE_REGEXP = re.compile(r'\s+')


def normalize_text(text):
    text = text.lower()
    text = re.sub(WHITESPACE_REGEXP, ' ', text)

    # remove accents: https://stackoverflow.com/a/518232
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = unicodedata.normalize('NFC', text)
    return text


def normalize_text_second(text, word_counter):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))  # 是个单词的集合
    # 另加
    temp = clean_str(text)
    words = temp.split()
    doc_words = []
    for word in words:
        # word not in stop_words and word_freq[word] >= 5
        if word not in stop_words and word_counter[word] >= 5:
            doc_words.append(word)

    text = ' '.join(doc_words).strip()
    # 另加
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


class Data(object):
    def __init__(self, name, instances, label_names):
        self.name = name
        self.instances = instances
        self.label_names = label_names

    def __iter__(self):
        for instance in self.instances:
            yield instance

    def __len__(self):
        return len(self.instances)

    def get_instances(self, fold=None):
        if fold is None:
            return self.instances
        else:
            return [ins for ins in self.instances if ins.fold == fold]


class DatasetInstance(object):
    def __init__(self, text, label, fold):
        self.text = text
        self.label = label
        self.fold = fold


class Dataset(object):
    def __init__(self, args):
        self.args = args

        self.fields = None
        self.data = None
        self.result = None

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

    def load_data(self, tokenizer, entity_linker, val_ratio=0.05):

        @functools.lru_cache(maxsize=None)
        def tokenize(text):
            return tokenizer.tokenize(text)

        @functools.lru_cache(maxsize=None)
        def detect_mentions(text):
            return entity_linker.detect_mentions(text)

        def create_numpy_sequence(source_sequence, length, dtype):
            ret = np.zeros(length, dtype=dtype)
            source_sequence = source_sequence[:length]
            ret[:len(source_sequence)] = source_sequence
            return ret

        # 生成一个统一的文件
        # 改变的主要为获取： train_data, test_data
        # 在这个文件里给 20Newsgroup 作成一个总的，在分配以后再给图

        dataset_20ng = REFORMAT_20NG(args=self.args)
        dataset_20ng.reformat_20news_bydate()
        train_data, test_data = dataset_20ng.partition_20news_bydate()

        val_size = int(len(train_data) * val_ratio)
        random.shuffle(train_data)

        print('---------------- fetch_20newsgroups 加载完成： ------------------')

        print(f'train_data 的长度为: {len(train_data)}')
        print(f'test_data 的长度为: {len(test_data)}')
        print(f"val_size 的长度为: {val_size}")

        instances = []
        instances += [DatasetInstance(text, label, 'val') for text, label in train_data[-val_size:]]
        instances += [DatasetInstance(text, label, 'train') for text, label in train_data[:-val_size]]
        instances += [DatasetInstance(text, label, 'test') for text, label in test_data]

        self.data = Data('20ng', instances, fetch_20newsgroups()['target_names'])

        word_counter = Counter()
        entity_counter = Counter()
        for instance in tqdm(self.data):
            word_counter.update(t.text for t in tokenize(instance.text))
            entity_counter.update(m.title for m in detect_mentions(instance.text))
        print(f'len(self.data_fed): {len(self.data)}')

        ########################################## 二次清洗 #############################################
        # for instance in self.data_fed:
        #     instance.text = normalize_text_second(instance.text, word_counter)
        #######################################################################################

        words = [word for word, count in word_counter.items() if count >= self.args.min_count]
        word_vocab = {word: index for index, word in enumerate(words, 1)}
        word_vocab[PAD_TOKEN] = 0
        print(f'len(word_vocab): {len(word_vocab)}')

        entity_titles = [title for title, count in entity_counter.items() if count >= self.args.min_count]
        entity_vocab = {title: index for index, title in enumerate(entity_titles, 1)}
        entity_vocab[PAD_TOKEN] = 0
        print(f'len(entity_vocab): {len(entity_vocab)}')

        self.result = dict(train=[], val=[], test=[], word_vocab=word_vocab, entity_vocab=entity_vocab)

        for fold in ('train', 'val', 'test'):
            for instance in self.data.get_instances(fold):
                word_ids = [word_vocab[token.text] for token in tokenize(instance.text) if token.text in word_vocab]
                entity_ids = []
                prior_probs = []
                for mention in detect_mentions(instance.text):
                    if mention.title in entity_vocab:
                        entity_ids.append(entity_vocab[mention.title])
                        prior_probs.append(mention.prior_prob)

                self.result[fold].append(dict(word_ids=create_numpy_sequence(word_ids, self.args.max_word_length, np.int),
                                              entity_ids=create_numpy_sequence(entity_ids, self.args.max_entity_length, np.int),
                                              prior_probs=create_numpy_sequence(prior_probs, self.args.max_entity_length, np.float32),
                                              label=instance.label))


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
