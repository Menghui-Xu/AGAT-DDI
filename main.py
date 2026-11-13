import os, pickle
import numpy
import argparse
import random
import csv
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
from model import Model, Model_avg, Model_sum, Model_cat, Model_hdm, Model_wo_feat, Model_wo_diffgraph, Model_wo_relgraph, Model_wo_att, Model_GAT, Model_GCN



'''Code for multi-class DDI prediction'''
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='which GPU card to use')
parser.add_argument('--hidden_dim', type=int, default=512, help='num of hidden features')
parser.add_argument('--out_dim', type=int, default=256, help='num of output features')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout rate')
parser.add_argument('--rels', type=int, default=65, help='num of relations')
parser.add_argument('--n_epochs', type=int, default=400, help='num of epochs')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--threshold', type=float, default=0.4, help='edge threshold')
parser.add_argument('--model', type=str, default='Model')
parser.add_argument('--pair_method', type=str, default='avg', help='avg or cat')
parser.add_argument('--filename', type=str, default='result.csv', help='folder to save results.csv')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--dataset', type=str, default='Deng')
args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


hidden_dim = args.hidden_dim
out_dim = args.out_dim
n_epochs = args.n_epochs
batch_size = args.batch_size
device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] using device: {device}')
print(f'dataset: {args.dataset}, seed: {args.seed}, fold: {args.fold}, results: {args.filename}')

# ============================
# 3. Load existing DDI splits
# ============================
train_df = pd.read_csv(f'{args.dataset}/{args.fold}/train.csv')
test_df = pd.read_csv(f'{args.dataset}/{args.fold}/test.csv')
train_edges = train_df[['Drug1', 'Drug2']].values
train_labels = train_df['Label'].values
test_edges = test_df[['Drug1', 'Drug2']].values
test_labels = test_df['Label'].values
print(f'Train: {len(train_edges)}, test: {len(test_edges)}')

# ============================
# 4. Load four attribute matrices
# ============================


def load_feat(similarity_profile_file):
    #读取特征数据
    init_feat = pd.read_csv(similarity_profile_file, header=None)
    #转换为NumPy数组
    init_feat = init_feat.to_numpy()
    return torch.FloatTensor(init_feat), torch.FloatTensor(init_feat)

feat_mat_chem, sim_mat_chem = load_feat(f'{args.dataset}/chem_Jacarrd_sim.csv')
feat_mat_target, sim_mat_target = load_feat(f'{args.dataset}/target_Jacarrd_sim.csv')
feat_mat_enzyme, sim_mat_enzyme = load_feat(f'{args.dataset}/enzyme_Jacarrd_sim.csv')
feat_mat_pathway, sim_mat_pathway = load_feat(f'{args.dataset}/pathway_Jacarrd_sim.csv')
train_feat_mat = torch.hstack([feat_mat_chem, feat_mat_target, feat_mat_enzyme, feat_mat_pathway])
train_feat_mat = torch.FloatTensor(train_feat_mat).to(device)
args.feat_dim = train_feat_mat.shape[1]
print(f'Attribute input dimension: {args.feat_dim}')


def get_edge_index(edge_list, edge_types):
    edge_list_image = edge_list[:, [1, 0]] # 生成反向边，将边 (u, v) 反向为 (v, u)，模拟无向图的双向关系
    edge_index = np.vstack([edge_list, edge_list_image]) # 合并原始边与反向边
    edge_index = torch.tensor(edge_index, dtype=torch.long)  # 将NumPy数组转为PyTorch张量，便于GPU计算。
    edge_type = np.concatenate([edge_types, edge_types])  #为反向边赋予与原始边相同的类型。
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    return edge_index.t().contiguous(), edge_type

def generate_edges_from_similarity(train_edges,  similarity_matrix,  threshold):
    # 从 train_edges 中提取所有训练集节点
    train_nodes = np.unique(train_edges.flatten()) #提取所有唯一的节点，并存储在 train_nodes 数组中。
    num_nodes = similarity_matrix.shape[0]
    new_edges = [] #初始化一个空列表 new_edges，用于存储新生成的边。

    train_nodes_set = set(train_nodes) #将训练集节点转换为集合，方便快速查找

    for i in range(num_nodes):
        # 仅处理训练集中的节点
        if i not in train_nodes_set:
            continue
        for j in range(i + 1, num_nodes):
            # 仅处理训练集中的节点
            if j not in train_nodes_set:
                continue
            if similarity_matrix[i, j] > threshold:#相似度超过阈值，则将这个节点对添加到 new_edges 列表中。
                new_edges.append([i, j])

    return np.array(new_edges, dtype=np.int64).reshape(-1, 2)


if args.threshold == 1:
    new_train_edges = train_edges
else:
    new_edges_chem = generate_edges_from_similarity(train_edges, sim_mat_chem.cpu().numpy(), args.threshold)
    new_edges_target = generate_edges_from_similarity(train_edges, sim_mat_target.cpu().numpy(), args.threshold)
    new_edges_enzyme = generate_edges_from_similarity(train_edges, sim_mat_enzyme.cpu().numpy(), args.threshold)
    new_edges_pathway = generate_edges_from_similarity(train_edges, sim_mat_pathway.cpu().numpy(), args.threshold)
    new_train_edges = np.vstack([train_edges, new_edges_chem, new_edges_target, new_edges_enzyme, new_edges_pathway])
    new_train_edges = np.sort(new_train_edges, axis=1)
    new_train_edges = np.unique(new_train_edges, axis=0)


class DDIDataset(Dataset):
    '''Customized dataset processing class'''

    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

train_dataset = DDIDataset(train_edges, train_labels)
test_dataset = DDIDataset(test_edges, test_labels)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

if args.model == 'Model':
    model = Model(args)
if args.model == 'sum':
    model = Model_sum(args)
if args.model == 'cat':
    model = Model_cat(args)
if args.model == 'avg':
    model = Model_avg(args)
if args.model == 'hdm':
    model = Model_hdm(args)
if args.model == 'feat':
    model = Model_wo_feat(args)
if args.model == 'dg':
    model = Model_wo_diffgraph(args)
if args.model == 'rg':
    model = Model_wo_relgraph(args)
if args.model == 'att':
    model = Model_wo_att(args)
if args.model == 'gat':
    model = Model_GAT(args)
if args.model == 'gcn':
    model = Model_GCN(args)

model.to(device)


def check_bidirectional(edges, name):
    s = set(map(tuple, edges))
    rev = sum(1 for (u, v) in s if (v, u) in s and u != v)
    print(f'[CHECK] {name}: {len(edges)} edges, bidirectional pairs = {rev // 2}')

check_bidirectional(train_edges, 'train_edges')
check_bidirectional(new_train_edges, 'new_train_edges')

# edge_index：主图的邻接关系，diff_edge_index：增广图的邻接关系，edge_type：边的类型信息，train_feat_mat：节点特征矩阵，edges：当前批次的边数据
edge_index, edge_type = get_edge_index(train_edges, train_labels)  # 构建图的邻接矩阵（可能包含边类型信息）
# print(edge_index.shape)
diff_edge_index, _ = get_edge_index(new_train_edges, train_labels)  # 构建图的邻接矩阵（可能包含边类型信息）
edge_index, edge_type, diff_edge_index = edge_index.to(device), edge_type.to(device), diff_edge_index.to(device)
print("主图边数:", edge_index.shape[1])
print("增广图边数:", diff_edge_index.shape[1])

def train_():
    criterion = nn.CrossEntropyLoss().to(device) #交叉熵损失函数，用于多分类问题
    re_criterion = nn.MSELoss().to(device)  #均方误差损失函数，评估重构输出（如自编码器）与原始输入的误差

    trainer_E = torch.optim.Adam(model.parameters(), lr=args.lr) #使用 Adam 优化器来更新模型参数，学习率为 1e-3

    n_iterations = len(train_loader) #每个 epoch 中的迭代次数，等于训练数据加载器的长度。
    num_epochs = n_epochs #总的训练轮数
    start = time.time() #记录训练开始时间

    running_loss = 0.0
    running_correct = 0.0

    # Training phase
    for epoch in range(num_epochs):
        model.train()
        true_labels, pred_labels = [], []
        running_loss = 0.0 #累计损失
        running_correct = 0.0 #累计正确分类的数量
        total_samples = 0 #总样本数


        for i, (edges, labels) in enumerate(train_loader):
            edges, labels = edges.to(device), labels.to(device)  # 迁移到GPU

            #y_pred：分类任务的预测（未归一化的概率），re_out：重构输出
            #print(edge_index[:10], diff_edge_index[:10], edge_type[:10], train_feat_mat[:10], edges[:10])
            y_pred, re_out = model(edge_index, diff_edge_index, edge_type, train_feat_mat, edges)


            # y_pred = model(edge_index, diff_edge_index, edge_type, train_feat_mat, edges)
            loss = criterion(y_pred, labels)
            re_loss = re_criterion(re_out, train_feat_mat)
            loss += re_loss

            trainer_E.zero_grad() # 清空历史梯度
            loss.backward() #通过计算图反向传播梯度（自动微分）
            trainer_E.step() #根据梯度方向更新参数（Adam自适应调整学习率）


            #torch.argmax(y_pred, dim=1)：找到概率最大的类别索引，即预测的类别。
            #torch.sum(...)：计算预测正确的样本总数，running_correct：累计正确分类的数量。
            running_correct += torch.sum((torch.argmax(y_pred, dim=1).type(torch.FloatTensor) == labels.cpu()).detach()).float()

            #将当前批次的预测标签添加到 pred_labels 列表中。
            pred_labels.append(list(y_pred.cpu().detach().numpy().reshape(-1)))

            labels = labels.cpu().numpy()
            #将当前批次的真实标签添加到 true_labels 列表中。
            true_labels.append(list(labels))

            #labels.shape[0]：获取当前批次的样本数量，total_samples += ...：累计总样本数量
            total_samples += labels.shape[0]
            #获取当前批次的损失值，并累计总损失
            running_loss += loss.item()

        print(
            f"epoch {epoch + 1}/{num_epochs};trainging loss: {running_loss / n_iterations:.4f} ;training set acc: {running_correct / total_samples:.4f}")

    test_(test_loader, epoch + 1)
    end = time.time() #结束训练的时间
    elapsed = end - start #训练时间统计
    print(f"Training completed in {elapsed // 60}m: {elapsed % 60:.2f}s.")

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    #计算多类分类问题的平均分数。
    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        # 确保输入为二维数组（n_samples × n_classes）
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        # 逐类别计算AUPR
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()  # 取第c类的真实标签
            y_score_c = y_score.take([c], axis=1).ravel()  # 取第c类的预测概率
            score[c] = binary_metric(y_true_c, y_score_c)    #计算该类AUPR
        return np.average(score)
    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)


def test_(loader, best_epoch):
    event_num = args.rels         # 类别的数量，这里假设有65个不同的事件或类别
    n_test_samples = 0     # 测试样本的总数
    n_correct = 0          # 正确预测的样本数
    total_labels = []      # 存储所有测试样本的真实标签
    total_pred = []        # 存储所有测试样本的预测概率
    # Testing phase
    with torch.no_grad():  # 确保在评估过程中不计算梯度，节省计算资源。
        model.eval()       # 将模型设置为评估模式
        for edges, labels in loader:

            edges, labels = edges.to(device), labels.to(device)

            ## 前向传播（仅计算分类输出） y_pred：分类任务的预测（未归一化的概率）
            y_pred, _ = model(edge_index, diff_edge_index, edge_type, train_feat_mat, edges)
            # y_pred= model(edge_index, diff_edge_index, edge_type, train_feat_mat, edges)
            #使用 softmax 函数将预测结果转换为概率分布
            y_hat = F.softmax(y_pred, dim=1)


            #将预测概率和真实标签从设备移动到CPU，并转换为NumPy数组。
            total_pred.append(y_hat.cpu().numpy())
            total_labels.append(labels.cpu().numpy())

            #计算测试样本总数，正确预测数
            n_test_samples += edges.shape[0]
            n_correct += torch.sum((torch.argmax(y_pred, dim=1).type(torch.FloatTensor) == labels.cpu()).detach()).float()


        total_pred = np.vstack(total_pred)  # 合并所有批次的预测概率
        total_labels = np.concatenate(total_labels)  # 合并真实标签
        pred_type = np.argmax(total_pred, axis=1)  # 获取预测类别 ([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]) # 预测概率
        y_one_hot = label_binarize(y=total_labels, classes=np.arange(event_num))  #([[0, 1], [1, 0], [0, 1]])  # 2个类别的真实标签


        acc = n_correct / n_test_samples  # 计算总体准确率。
        # y_true = np.array([[0, 1], [1, 0], [0, 1]])  # 2个类别的真实标签
        # y_score = np.array([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]) # 预测概率
        aupr = roc_aupr_score(y_one_hot, total_pred, average='micro')
        auroc = roc_auc_score(y_one_hot, total_pred, average='micro')
        f1 = f1_score(total_labels, pred_type, average='macro')
        precision = precision_score(total_labels, pred_type, average='macro', zero_division=0)
        recall = recall_score(total_labels, pred_type, average='macro', zero_division=0)

        print(f"-------test set result---------")
        print(f"ACC: {acc}")
        print(f"AUPR: {aupr}")
        print(f"AUROC: {auroc}")
        print(f"F1: {f1}")
        print(f"Precison: {precision}")
        print(f"Recall: {recall}")
        # print(f"{acc}\t{aupr}\t{auroc}{f1}\t{precision}\t{recall}")
        save_results(best_epoch, acc, aupr, auroc, f1, precision, recall)
        return f1

def save_results(best_epoch, acc, aupr, auroc, f1, precision, recall):
    try:
        with open(args.filename, 'r') as f:
            pass
    except FileNotFoundError:
        with open(args.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'dataset', 'seed', 'fold', 'batch_size', 'n_epochs',
                             'threshold', 'pair_method', 'dropout', 'lr',
                             'ACC', 'AUPR', 'AUROC', 'F1', 'Precision', 'Recall'])
    with open(args.filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([args.model, args.dataset, args.seed, args.fold, args.batch_size, best_epoch,
                         args.threshold, args.pair_method, args.dropout, args.lr,
                         f'{acc:.6f}', f'{aupr:.6f}', f'{auroc:.6f}',
                         f'{f1:.6f}', f'{precision:.6f}', f'{recall:.6f}'])


train_()
