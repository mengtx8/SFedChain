import argparse
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='agnews')  # ['20ng', 'agnews', 'r8']
    # r8

    parser.add_argument('--num_nodes', type=int, default=7)
    parser.add_argument('--num_users', type=int, default=7, help="number of users: K")
    parser.add_argument('--max_nums', type=int, default=6)
    parser.add_argument('--num_max_nums', type=int, default=6)

    # 窗口 5
    # nohup python -Ru main_fedweight.py > result.log_1 2>&1 &
    parser.add_argument('--gpu', type=int, default=3, help="GPU ID, -1 for CPU")
    parser.add_argument('--val_ratio_global', type=float, default=0.025)
    parser.add_argument('--val_ratio_local', type=float, default=0.04)











    # federated arguments

    parser.add_argument('--model', type=str, default='NABoE', help='model name')




    # 节点数共9个，将20ng的数据划分为9份，
    # 划分为两份的情况下，两个用户参加。。。
    # 划分为三份的情况下，三个用户参加。。。
    # 划分为四份的情况下，四个用户参加。。。
    # 划分为五份的情况下，五个用户参加。。。
    # num_nodes固定不变，然后调num_users，从两个--三个--四个
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--wikipedia2vec_file', type=str, default='enwiki_20180420_lg1_300d.pkl')
    parser.add_argument('--entity_linker_file', type=str, default='enwiki_20180420_entity_linker.pkl')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--use_word', type=bool, default=True)

    # 20ng
    parser.add_argument('--min_count', type=int, default=3)
    parser.add_argument('--max_word_length', type=int, default=64)
    parser.add_argument('--max_entity_length', type=int, default=256)
    # parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patience_local', type=int, default=30)
    parser.add_argument('--patience_global', type=int, default=30)
    parser.add_argument('--learning_rate', type=int, default=1e-3)
    parser.add_argument('--weight_decay', type=int, default=0.1)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--dropout_prob', type=int, default=0.5)
    parser.add_argument('--iid', type=bool, default=True, help='whether i.i.d or not')
    parser.add_argument('--all_clients', type=bool, default=True, help='aggregation over all clients')

    args = parser.parse_args()
    return args
