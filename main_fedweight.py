from d2l import torch as d2l
import torch
import copy
import time
import wikipedia2vec.utils.tokenizer.regexp_tokenizer
import entity_linker
from torch.utils.data import DataLoader

import sys
sys.path.append('/home/user/Documents/mmm/paper/wikipedia2vec-master/examples/fed_avg')
import data_fed.utils_data_fed
import utils_fed.update
from utils_fed.fedavg import *
import models.naboe
from utils_fed.options import args_parser
from utils_fed.sampling import *
# from utils_fed.update import *
# from utils_fed.utils_fed import *
# from utils_fed.fedavg import *
# from utils_fed.evaluate import *


class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.tik = None

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)

if __name__ == '__main__':
    timer = Timer()

    args = args_parser()
    embedding = wikipedia2vec.Wikipedia2Vec.load(args.wikipedia2vec_file)
    tokenizer = wikipedia2vec.utils.tokenizer.regexp_tokenizer.RegexpTokenizer()
    entity_linker = entity_linker.EntityLinker(args.entity_linker_file)

    print('---------------- 打印实验信息： ------------------')
    print(f'数据集：{args.dataset}')
    print(f'节点数：{args.num_nodes}')
    print(f'用户数：{args.num_users}')
    print(f'模型：{args.model}')
    print(f'val_ratio_global：{args.val_ratio_global}')
    print(f'val_ratio_local：{args.val_ratio_local}')
    print(f'batch_size：{args.batch_size}')
    print(f'GPU：{args.gpu}')

    # parse args
    global dataset
    global dict_users
    global global_model
    global word_vocab
    global entity_vocab
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # load dataset and split users  -->  dataset_train, dataset_test
    if args.dataset == '20ng':
        dataset = data_fed.utils_data_fed.Dataset(args)
        dataset.load_data(tokenizer, entity_linker, val_ratio=args.val_ratio_global)
    elif args.dataset == 'agnews':
        dataset = data_fed.utils_data_fed.Dataset(args)
        dataset.load_data(tokenizer, entity_linker, val_ratio=args.val_ratio_global)
    elif args.dataset == 'r8':
        dataset = data_fed.utils_data_fed.Dataset(args)
        dataset.load_data(tokenizer, entity_linker, val_ratio=args.val_ratio_global)
    else:
        exit('main_fedweight_backups.py: line: 74')

    word_vocab = dataset.result['word_vocab']
    entity_vocab = dataset.result['entity_vocab']
    print('---------------- dataset加载完成： ------------------')
    print(f'dataset.result["train"] 的长度为: {len(dataset.result["train"])}')
    print(f'dataset.result["val"] 的长度为: {len(dataset.result["val"])}')
    print(f'dataset.result["test"] 的长度为: {len(dataset.result["test"])}')
    # while True: pass
    # sample users
    if args.iid:
        dict_users = mnist_iid(dataset.result['train'], args.num_users)
        print('---------------- 各节点数据字典： ------------------')
    else:
        pass
    ######################  word_embedding  entity_embedding  ###############################################
    dim_size = embedding.syn0.shape[1]
    word_embedding = np.random.uniform(low=-0.05, high=0.05, size=(len(word_vocab), dim_size))
    word_embedding[0] = np.zeros(dim_size)
    for word, index in word_vocab.items():
        try:
            word_embedding[index] = embedding.get_word_vector(word)
        except KeyError:
            continue
    entity_embedding = np.random.uniform(low=-0.05, high=0.05, size=(len(entity_vocab), dim_size))
    entity_embedding[0] = np.zeros(dim_size)
    for entity, index in entity_vocab.items():
        try:
            entity_embedding[index] = embedding.get_entity_vector(entity)
        except KeyError:
            continue

    val_iterator_global = DataLoader(dataset.result['val'], shuffle=False, batch_size=args.batch_size)
    test_iterator_global = DataLoader(dataset.result['test'], shuffle=False, batch_size=args.batch_size)
    #####################################################################

    # build model  -->  net_glob
    if args.model == 'NABoE' and args.dataset == '20ng':
        print('---------------- 模型为： ------------------')
        print("args.model == 'NABoE' and args.dataset == '20ng'")
    elif args.model == 'NABoE' and args.dataset == 'agnews':
        print('---------------- 模型为： ------------------')
        print("args.model == 'NABoE' and args.dataset == 'agnews'")
    elif args.model == 'NABoE' and args.dataset == 'r8':
        print('---------------- 模型为： ------------------')
        print("args.model == 'NABoE' and args.dataset == 'r8'")
    else:
        exit('Error: unrecognized model')
    global_model = models.naboe.NABoE(word_embedding, entity_embedding, len(dataset.data.label_names), args.dropout_prob, args.use_word)

    print(global_model)
    print('model_global打印成功！')
    global_model.to(args.device)
    global_model.train()

    # copy weights  -->  w_glob
    w_global = global_model.state_dict()
    print('\w_global输出成功！')

    # training
    loss_train = []
    global w_locals
    global local_data_s

    if args.all_clients:
        m = args.num_users
        print(f"Aggregation over clients: {m}, all vlients")
        w_locals = [w_global for i in range(args.num_users)]
    else:
        print("错误")
        # m = max(int(args.frac * args.num_users), 1)
        # pri-==cnt(f"Aggregation over clients: {m}")
        # w_locals = []
    idx_users = np.random.choice(range(args.num_users), m, replace=False)
    print(idx_users)

    epoch_global = 0
    best_val_acc_global = 0.0
    best_val_acc_test_acc_global = 0.0
    best_weights_global = None
    num_epochs_without_improvement_global = 0
    time_list = []
    acc_local_num_users = []  # 每次迭代，各本地模型的准确率组成的列表
    # while True:
    for i in range(30):
        epoch_global += 1
        # if epoch_global > 30:
        #     break
        print(f'\033[41m###############  Global Aggregation: {epoch_global}  ################\033[0m')
        print(f'\033[41m#############  连续 验证集准确率 无增长次数：{num_epochs_without_improvement_global}  #############\033[0m')
        local_losses = []
        local_data_s = []
        for i, idx in enumerate(idx_users):  # 对每个用户进行训练, 每个本地数据都有一个深拷贝的dataset
            print(f'********** 第{i+1}个节点，共{len(idx_users)}个节点 **********')
            print(f"Local Train idx: {idx}")
            local_model = utils_fed.update.LocalUpdate(model=copy.deepcopy(global_model).to(args.device),
                                                       args=args, dataset=dataset, dict_users=dict_users, idx=idx,
                                                       embedding=embedding, tokenizer=tokenizer, entity_linker=entity_linker,
                                                       val_ratio=args.val_ratio_local)
            # timer.start()  # 本地模型计时开始
            w, loss, local_data = local_model.train(timer, acc_local_num_users)
            # timer.stop()  # 本地模型计时结束
            if epoch_global == 2:
                print(timer.times[:8])
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_data_s.append(local_data[2])
        time_list.append(max(timer.times[-args.num_users:]))  # 取本次本地模型训练时间的最大值

        # update global weights  -->  平均聚合与加权聚合
        # timer.start()  # 全局模型计时开始
        w_global = fedrated_avg(w_locals, local_data_s)
        # timer.stop()  # 全局模型计时结束
        # time_list.append(timer.times[-1])
        # copy weight to net_glob
        global_model.load_state_dict(w_global)

        # print loss
        loss_avg = sum(local_losses) / len(local_losses)
        print(f'\033[35m全局聚合, Average loss {loss_avg:.3f}')
        # loss_train.append(loss_avg)
        # Global Aggregation Evaluate
        val_acc_global, val_acc_global_f1 = round(utils_fed.update.evaluate(global_model, val_iterator_global, args.device, 'val')[0], 4), \
                                            round(utils_fed.update.evaluate(global_model, val_iterator_global, args.device, 'val')[0], 4)
        test_acc_global, test_acc_global_f1 = utils_fed.update.evaluate(global_model, test_iterator_global, args.device, 'test')
        if val_acc_global >= best_val_acc_global:
            best_val_acc_global = val_acc_global
            best_val_acc_test_acc_global = test_acc_global
            best_weights_global = {k: v.to('cpu').clone() for k, v in global_model.state_dict().items()}
            num_epochs_without_improvement_global = 0
        else:
            num_epochs_without_improvement_global += 1

        # if num_epochs_without_improvement_global >= args.patience_global:
        #     global_model.load_state_dict(best_weights_global)
        #     break
        print('**************** 本轮次聚合结果： ****************')
        print('\033')
        print(f'全局聚合, Evaluate global val Accuracy: {val_acc_global:.4f}, Evaluate global val Accuracy_f1: {val_acc_global_f1}')
        print(f'全局聚合, Evaluate global Test Accuracy: {test_acc_global:.4f}, Evaluate global val Accuracy_f1: {test_acc_global_f1}')
        print(f'最好的 验证集准确率：{best_val_acc_global}, 此时刻 测试集准确率：{best_val_acc_test_acc_global}     \033[0m')
    global_model.load_state_dict(best_weights_global)

    # 最终训练准确率
    print('---------------- 打印实验信息： ------------------')
    print('\033')
    print(f'数据集：{args.dataset}')
    print(f'节点数：{args.num_nodes}')
    print(f'用户数：{args.num_users}')
    print(f'max_nums：{args.max_nums}')
    print(f'num_max_nums：{args.num_max_nums}')
    print(f'模型：{args.model}')
    print(f'val_ratio_global：{args.val_ratio_global}')
    print(f'val_ratio_local：{args.val_ratio_local}')
    print(f'batch_size：{args.batch_size}')
    print(f'GPU：{args.gpu}')

    test_acc_global = utils_fed.update.evaluate(global_model, test_iterator_global, args.device, 'test')[0]
    print(f'全局模R型建模准确率（Evaluate global Test Accuracy）: {test_acc_global:.4f}')
    print(f'本地模型建模准确率均值（Evaluata local Test Accuracy）: {sum(acc_local_num_users[:args.num_users]) / args.num_users}')
    print(f'运行结束时间：{time.strftime("% Y-%m-%d-%H_%M_%S", time.localtime())}')
    print(f'本地模型建模时间均值为（毫秒级时间）：{sum(timer.times[:args.num_users]) / args.num_users}ms')
    print(f'联合模型建模时间（毫秒级时间）：{int(round(sum(time_list) * 1000))}ms')  # f'联合模型建模时间（秒级时间）：{int(sum(time_list))}s')  # 秒级时间戳
    print(f'len(timer.times):{len(timer.times)}')
    # print(acc_local_num_users[:args.num_users])

