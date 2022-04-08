import os
import random
import re
import shutil
import pandas as pd

import nltk
import numpy as np
import torch
import sys

from nltk.corpus import stopwords
from tqdm import tqdm

sys.path.append('/home/user/Documents/mmm/paper/wikipedia2vec-master/examples/fed_avg/utils_fed')
import options

# filepath = '20news-bydate-train'
path_fed_avg = '/home/user/Documents/mmm/paper/wikipedia2vec-master/examples/fed_avg'
path_data = '/home/user/Documents/mmm/paper/wikipedia2vec-master/examples/fed_avg/data_fed'
path_agnews = '/home/user/Documents/mmm/paper/wikipedia2vec-master/examples/fed_avg/data_fed/agnews'


# filepath = os.path.join(path, '20news-bydate/20news-bydate-train')

# 所有文件的路径集合，每行三列，example:
# 'data_fed/20news-bydate-train/comp.os.ms-windows.misc/9537	20news-bydate-train	comp.os.ms-windows.misc'

class REFORMAT_AGNEWS:
    def __init__(self, args):
        """ 三种列表已经集齐：
        所有test和train内容列表：   self.all_content_list = []
        test内容列表:   self.test_content_list = []
        train内容列表:   self.train_content_list = []
        文件路径列表：     file_paths = []
        content_list 列表：   content_list = []
        原始主题列表：         primitive_topic_list = []
        数字（ID）主题列表：     number_topic_list = []
        """
        self.args = args
        self.test_all_content_list = []
        self.train_all_content_list = []

        self.textgcn_path_list = []
        self.textgcn_content_list = []
        self.train_data = []
        self.test_data = []

    def reformat_agnews(self):
        for name in ['train', 'test']:
            filepath = os.path.join(path_agnews, 'ag_news_csv', name + '.csv')
            print(f'操作 agnews 数据：{filepath}')
            # 文件路径列表，content_list 列表
            file_paths_list = []
            content_list = []

            # Load Data
            data = pd.read_csv(filepath)

            # Set Column Names
            data.columns = ['ClassIndex', 'Title', 'Description']

            # Combine Title and Description
            x_com_train = data['Title'] + " " + data['Description']  # Combine title and description (better accuracy than using them as separate features)
            x_com_train = x_com_train.map(lambda x: x.strip())
            y_com_train = data['ClassIndex'].apply(lambda x: x)  # Class labels need to begin from 0
            full_df_train = pd.DataFrame({"label": y_com_train, "text": x_com_train})

            for i, data in enumerate(full_df_train.values):
                file_path = os.path.join('data_fed', 'ag_news_csv', name, str(i) + '.txt')
                file_paths_list.append(file_path.strip() + '\t' + name + '\t' + str(data[0]))
                content_list.append(data[1])

            # 原始主题列表
            primitive_topic_list = []
            for file_path in file_paths_list:
                primitive_topic_list.append(file_path.split('\t')[-1])
            primitive_topic_set = set(primitive_topic_list)  # 文档主题集合
            print(f'topic_set, size: {len(primitive_topic_set)}')
            topic_dic = {}  # 文档主题字典
            for i, doc_topic in enumerate(primitive_topic_set):
                topic_dic[doc_topic] = i


            # 三种列表已经集齐：
            #         文件路径列表：     file_paths = []
            #         content_list 列表：   content_list = []
            #         原始主题列表：         primitive_topic_list = []
            print(file_paths_list[0])
            print(file_paths_list[1])
            print(file_paths_list[2])
            print(content_list[0])
            print(content_list[1])
            print(content_list[2])
            print(primitive_topic_set)
            print(f'file_paths_list长度为：{len(file_paths_list)}, \t content_list长度为：{len(content_list)} \t'
                  f'primitive_topic_list长度为：{len(primitive_topic_list)}')
            if name == 'test':
                # 总内容列表： all_content_list = []
                # eg：('data_fed/20ng/20news-bydate-train/comp.os.ms-windows.misc/9574\t20news-bydate-train\tcomp.os.ms-windows.misc', '0', "b'From")
                for i in range(len(file_paths_list)):
                    self.test_all_content_list.append((file_paths_list[i], content_list[i]))
                print(f'{name}文件的长度为：{len(self.test_all_content_list)}')
                print('test_all_content_list 样本如下：')
                print(self.test_all_content_list[0])
            elif name == 'train':
                for i in range(len(file_paths_list)):
                    self.train_all_content_list.append((file_paths_list[i], content_list[i]))
                print(f'{name}文件的长度为：{len(self.train_all_content_list)}')
                random.shuffle(self.train_all_content_list)  # 对训练数据进行打乱
                print('train_all_content_list 样本如下：')

    def partition_agnews(self):
        # 根据参与 fed 的 num_nodes 进行 20ng 数据的划分，有几个 num_nodes 就划分为多少份数据
        num_items = int(len(self.train_all_content_list) / self.args.num_nodes)
        partitioned_train_all_content_list_users = random.sample(self.train_all_content_list,
                                                                 num_items * self.args.num_users)
        partitioned_test_all_content_list = self.test_all_content_list

        # 将总数据分给 num_nodes 个节点（每个 node 表示一个 user），有 num_users 个用户参与，每个 user 有 num_eights 数据参与
        num_items_user = int(len(partitioned_train_all_content_list_users) / self.args.max_nums)  # 每个用户的数据最多分为 6 份
        # 分配给每个用户的数据数：len(partitioned_train_all_content_list) / self.args.num_users
        partitioned_train_all_content_list = random.sample(partitioned_train_all_content_list_users,
                                                           num_items_user * self.args.num_max_nums)
        print('##################################################################################')
        print(f'总训练数据数：{len(self.train_all_content_list)}')
        print(
            f'int({len(self.train_all_content_list)} / {self.args.num_nodes}) = {int(len(self.train_all_content_list) / self.args.num_nodes)}, '
            f'int({len(partitioned_train_all_content_list_users)} / self.args.max_nums) = {int(len(partitioned_train_all_content_list_users) / self.args.max_nums)}')
        print(f"节点总数：{self.args.num_nodes}，每个节点拥有的数据数：{num_items} \n"
              f"用户总数：{self.args.num_users}，每个用户拥有的数据数：{num_items} \n"
              f"理论(num_eights)下，利用每个用户的数据数：{int(num_items * (self.args.num_max_nums / self.args.max_nums))} \n"
              f"实际分配给用户的总数据数：{int(len(partitioned_train_all_content_list) / self.args.num_users)}")
        print('分配数据示例：')
        print(partitioned_train_all_content_list[0])
        print('##################################################################################')

        # 做好了：partitioned_train_all_content_list， partitioned_test_all_content_list

        # 交给 "textGcn" 的文件，有几个 num_users 就分配多少 num_items * self.args.num_users 数据
        print('---------------- 交给 "textGcn" 的文件, 两个文件 ./corpus/20ng.txt 和 ./20ng.txt ------------------')
        for partition_tt_content_list in [partitioned_train_all_content_list, partitioned_test_all_content_list]:
            for line in partition_tt_content_list:
                self.textgcn_path_list.append(line[0] + '\n')
                self.textgcn_content_list.append(line[1] + '\n')
        print(len(self.textgcn_path_list))
        print(len(self.textgcn_content_list))

        with open(os.path.join(path_agnews, 'agnews.txt'), 'w') as f:
            for line in self.textgcn_path_list:
                f.write(str(line))
        with open(os.path.join(path_agnews, 'corpus', 'agnews.txt'), 'w') as f:
            for line in self.textgcn_content_list:
                f.write(str(line))

        with open(os.path.join(path_agnews, 'agnews.txt'), 'r') as f:
            print(f'os.path.join(path, "agnews.txt"): {len(f.readlines())}')
        with open(os.path.join(path_agnews, 'corpus', 'agnews.txt'), 'r') as f:
            print(f'os.path.join(path, "corpus", "agnews.txt"): {len(f.readlines())}')
        self.copy_file(os.path.join(path_agnews, 'agnews.txt'), os.path.join(path_fed_avg, 'text_gcn/data/agnews.txt'))
        self.copy_file(os.path.join(path_agnews, 'corpus', 'agnews.txt'),
                       os.path.join(path_fed_avg, 'text_gcn/data/corpus/agnews.txt'))

        # 测试 'text_gcn/data/agnews.txt'
        print('测试 text_gcn/data/agnews.txt')
        with open(os.path.join(path_fed_avg, 'text_gcn/data/agnews.txt'), 'r') as f:
            print(f'os.path.join(path, "agnews.txt"): {len(f.readlines())}')
        with open(os.path.join(path_fed_avg, 'text_gcn/data/corpus/agnews.txt'), 'r') as f:
            print(f'os.path.join(path, "corpus", "agnews.txt"): {len(f.readlines())}')

        # 交给 naboe 的文件，有几个 num_users 就分配多少 num_items * self.args.num_users 数据
        print('---------------- 交给 "naboe" 的文件，一个文件 ------------------')
        print(partitioned_train_all_content_list[0])
        for all_content_tuple in partitioned_train_all_content_list:
            self.train_data.append((all_content_tuple[1], int(all_content_tuple[0].split('\t')[2])))  # (text, label)
        for all_content_tuple in partitioned_test_all_content_list:
            self.test_data.append((all_content_tuple[1], int(all_content_tuple[0].split('\t')[2])))  # (text, label)
        print(len(self.train_data))
        print(len(self.test_data))
        print(len(self.train_data) + len(self.test_data))
        # print(self.train_data[0])

        # self.train_data, self.test_data = self.clean_data(self.train_data, self.test_data)
        print('88888888')
        print('\t\t\t', len(self.train_data))
        print('\t\t\t', len(self.test_data))
        # print(self.train_data[0])

        # 对训练数据进行清洗
        return self.train_data, self.test_data, partitioned_train_all_content_list, partitioned_test_all_content_list

    def clean_data(self, train_data, test_data):
        # 不对训练数据进行清理
        print('对测试数据进行了清理')
        tqdm(nltk.download('stopwords'))
        stop_words = set(stopwords.words('english'))
        len_train_data, len_test_data = len(train_data), len(test_data)

        test_content_list = []  # 前是 train，后是 test
        for line in test_data:
            test_content_list.append(line[0].strip())

        # doc_content_list = []  # 前是 train，后是 test
        # for doc_content in [train_data, test_data]:
        #     for line in doc_content:
        #         doc_content_list.append(line[0].strip())

        word_freq = {}  # to remove rare words
        for test_content in test_content_list:
            temp = self.clean_str(test_content)
            words = temp.split()
            for word in words:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1

        clean_docs = []
        for doc_content in test_content_list:
            temp = self.clean_str(doc_content)
            words = temp.split()
            doc_words = []
            for word in words:
                # word not in stop_words and word_freq[word] >= 5
                if self.args.dataset == 'mr':
                    doc_words.append(word)
                elif word not in stop_words and word_freq[word] >= 2:
                    doc_words.append(word)
            doc_str = ' '.join(doc_words).strip()
            clean_docs.append(doc_str)

        clean_train_data, clean_test_data = [], []
        for line in train_data[:]:
            clean_train_data.append(tuple((line[0], line[1])))
        for i, line in enumerate(clean_docs[:]):
            clean_test_data.append(tuple((line, test_data[i][1])))
        return clean_train_data, clean_test_data

    def clean_str(self, string):
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

    # source_file:源路径, target_ir:目标路径
    def copy_dir(self, source_dir, target_dir):
        if os.path.exists(target_dir):
            if len(os.listdir(target_dir)) == 0:
                os.rmdir(target_dir)

        for file in os.listdir(source_dir):
            source_file = os.path.join(source_dir, file)
            if os.path.isfile(source_file):
                shutil.copy(source_file, target_dir)

    # source_file:源路径, target_ir:目标路径
    def copy_file(self, source_file, target_file):
        if os.path.exists(target_file):
            os.remove(target_file)
        with open(target_file, 'a') as f:
            pass

        shutil.copyfile(source_file, target_file)


# 测试
if __name__ == '__main__':
    args = options.args_parser()
    dataset = REFORMAT_AGNEWS(args=args)
    dataset.reformat_agnews()
    dataset.partition_agnews()
