import os

import torch
from torch import nn
from torchtext import data
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import sys

sys.path.append('/home/user/Documents/mmm/paper/wikipedia2vec-master/examples/fed_avg')
from torch.utils.data import DataLoader
from utils_fed.optimizer import *


# model_global.add_optimizer(optim.SGD(model_global.parameters(), lr=args.lr))
# model_global.add_loss_op(nn.NLLLoss())
# optimizer = optim.SGD(model_global.parameters(), lr=args.lr)
#
# train_losses = []
# val_accuracies = []

class DatasetLocal:
    def __init__(self, dataset, idx_value):
        self.dataset = dataset
        self.idx_value = list(idx_value)

        self.local_result = dict(train=[], val=[],
                                 test=dataset.result['test'],
                                 word_vocab=dataset.result['word_vocab'],
                                 entity_vocab=dataset.result['entity_vocab'])

        self.local_train_examples = []
        self.local_dataset_train = []
        self.local_dataset_test = None
        self.local_dataset_val = None
        self.local_test_iterator = None

    def local_load_data(self, val_ratio=None):
        for i, value in enumerate(self.dataset.result['train']):
            if i in self.idx_value:
                self.local_result['train'].append(value)
        if val_ratio is not None:
            print('val_ratio 不空')
            val_size = int(len(self.local_result['train']) * val_ratio)
            self.local_result['train'] = [cut for cut in self.local_result['train'][:-val_size]]
            self.local_result['val'] = [cut for cut in self.local_result['train'][-val_size:]]


class LocalUpdate(object):
    def __init__(self, model, args, dataset, dict_users, idx, embedding, tokenizer, entity_linker, val_ratio=None):
        self.model = model
        self.args = args
        self.dict_users = dict_users
        self.idx = idx
        self.idx_value = self.dict_users[self.idx]
        self.embedding = embedding,
        self.tokenizer = tokenizer,
        self.entity_linker = entity_linker,
        self.val_ratio = val_ratio

        self.loss = None
        self.optimizer = None

        # def load_local_data():  // 对dataset进行本地分割
        dataset_local = DatasetLocal(dataset, self.dict_users[self.idx])
        dataset_local.local_load_data(val_ratio=self.val_ratio)
        print(f'\t节点 "train" 长度：{len(dataset_local.local_result["train"])}')
        print(f'\t节点 "val" 长度：{len(dataset_local.local_result["val"])}')
        print(f'\t节点 "test" 长度：{len(dataset_local.local_result["test"])}')
        # 将本地训练数据的大小传给 textgcn
        path_20ng = '/home/user/Documents/mmm/paper/wikipedia2vec-master/examples/fed_avg/data_fed/20ng'
        path_agnews = '/home/user/Documents/mmm/paper/wikipedia2vec-master/examples/fed_avg/data_fed/agnews'
        path_r8 = '/home/user/Documents/mmm/paper/wikipedia2vec-master/examples/fed_avg/data_fed/r8'
        if self.args.dataset == '20ng':
            with open(os.path.join(path_20ng, 'num_train.txt'), 'w') as f:
                f.write(str(len(dataset_local.local_result["train"]) + len(dataset_local.local_result["val"])))
        elif self.args.dataset == 'agnews':
            with open(os.path.join(path_agnews, 'num_train.txt'), 'w') as f:
                f.write(str(len(dataset_local.local_result["train"]) + len(dataset_local.local_result["val"])))
        elif self.args.dataset == 'r8':
            with open(os.path.join(path_r8, 'num_train.txt'), 'w') as f:
                f.write(str(len(dataset_local.local_result["train"]) + len(dataset_local.local_result["val"])))
        else:
            print('错误！')
            exit("update.py line:88 -> 将本地训练数据的大小传给 textgcn")

        self.local_result = dataset_local.local_result

        self.local_train_iterator = DataLoader(self.local_result['train'], shuffle=True, batch_size=self.args.batch_size)
        self.local_val_iterator = DataLoader(self.local_result['val'], shuffle=False, batch_size=self.args.batch_size)
        self.local_test_iterator = DataLoader(self.local_result['test'], shuffle=False, batch_size=self.args.batch_size)

    def train(self, timer=None, acc_local_num_users=None):  # 对每个用户进行训练
        self.model.train()
        # train and update
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay,
                               warmup=self.args.warmup_epochs * len(self.local_train_iterator))
        self.model.to(self.args.device)

        epoch = 0
        best_val_acc = 0.0
        best_weights = None
        num_epochs_without_improvement = 0

        epoch_loss = []
        if self.val_ratio is not None:
            timer.start()  # 本地模型计时开始
            # while True:
            for i in range(30):
                epoch += 1
                if epoch > 30:
                    break
                batch_loss = []
                self.model.train()
                for batch in self.local_train_iterator:
                    args_batch = {k: v.to(self.args.device) for k, v in batch.items() if k != 'label'}
                    logits = self.model(**args_batch)
                    loss = F.cross_entropy(logits, batch['label'].to(self.args.device))
                    # loss = F.cross_entropy(logits, torch.tensor(batch['label']))
                    loss.backward()
                    self.optimizer.step()
                    self.model.zero_grad()
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))


                val_acc = evaluate(self.model, self.local_val_iterator, self.args.device, 'val')[0]
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = {k: v.to('cpu').clone() for k, v in self.model.state_dict().items()}
                    num_epochs_without_improvement = 0
                else:
                    num_epochs_without_improvement += 1

                    # if num_epochs_without_improvement >= self.args.patience_local:
                    #     self.model.load_state_dict(best_weights)
                    #     break
            self.model.load_state_dict(best_weights)
            timer.stop()  # 本地模型计时结束
            # return evaluate(model, test_data_loader, self.device, 'test')

            local_train_acc, local_train_f1 = evaluate(self.model, self.local_train_iterator, self.args.device, 'test')
            local_val_acc, local_val_f1 = evaluate(self.model, self.local_val_iterator, self.args.device, 'test')
            local_test_acc, local_test_f1 = evaluate(self.model, self.local_test_iterator, self.args.device, 'test')
            acc_local_num_users.append(local_test_acc)
            # print(f'\tLocal Train idx: {self.idx}, Evaluate Training Accuracy: {local_train_acc:.4f}')
            # print(f'\tLocal Train idx: {self.idx}, Evaluate Validation Accuracy: {local_val_acc:.4f}')
            # print(f'\tLocal Train idx: {self.idx}, Evaluate Test Accuracy: {local_test_acc:.4f}')
            print('********************')

            return self.model.state_dict(), \
                   sum(epoch_loss) / len(epoch_loss), \
                   (local_train_acc, local_val_acc, local_test_acc)
        else:
            print("self.val_ratio is None 错误！！！！！！！！！！！！！！！！！！！！！")


def evaluate(model, data_loader, device, fold):
    model.eval()

    predictions = []
    labels = []
    with torch.no_grad():
        for batch in data_loader:
            args = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            logits = model(**args)
            predictions += torch.argmax(logits, 1).to('cpu').tolist()
            labels += batch['label'].to('cpu').tolist()

    test_acc = accuracy_score(labels, predictions)
    # print(f'accuracy ({fold}): {test_acc:.4f}')

    test_f1 = f1_score(labels, predictions, average='macro')
    # print(f'f-measure ({fold}): {test_f1:.4f}')

    return test_acc, test_f1
