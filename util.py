import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(dataset, degree_as_tag, verbose=True):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    if dataset in ['COX2', 'DD', 'ENZYMES']:
        from torch_geometric.datasets import TUDataset
        from torch_geometric.utils import to_networkx, degree
        dataset_obj = TUDataset(root='dataset/', name=dataset)
        g_list = []
        for data in dataset_obj:
            g_nx = to_networkx(data)
            g_list.append(S2VGraph(g_nx, data.y.item(), node_tags=None))

        for i, g in enumerate(g_list):
            g.edge_mat = dataset_obj[i].edge_index
            g.node_features = dataset_obj[i].x
            N = g.node_features.size(0)
            g.max_neighbor = degree(g.edge_mat[0], num_nodes=N)

            g.neighbors = [[] for i in range(N)]
            for (i, j) in g.edge_mat.T:
                g.neighbors[i].append(j)
        return g_list, dataset_obj.num_classes
    elif dataset in ['REDDIT-MULTI-5K']:
        from torch_geometric.datasets import TUDataset
        from torch_geometric.utils import to_networkx, degree
        dataset_obj = TUDataset(root='dataset/', name=dataset)
        g_list = []
        for data in dataset_obj:
            g_nx = to_networkx(data)
            g_list.append(S2VGraph(g_nx, data.y.item(), node_tags=None))

        for i, g in enumerate(g_list):
            g.edge_mat = dataset_obj[i].edge_index
            
            # 检查 node_features 是否存在
            if dataset_obj[i].x is not None:
                g.node_features = dataset_obj[i].x
            else:
                # 初始化默认节点特征 (这里以全1特征向量为例)
                num_nodes = dataset_obj[i].num_nodes
                g.node_features = torch.ones((num_nodes, 1))
            
            N = g.node_features.size(0)
            g.max_neighbor = degree(g.edge_mat[0], num_nodes=N)

            g.neighbors = [[] for i in range(N)]
            for (i, j) in g.edge_mat.T:
                g.neighbors[i].append(j)

        return g_list, dataset_obj.num_classes
    elif dataset in ['ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp']:
        from ogb.graphproppred import PygGraphPropPredDataset
        from torch_geometric.utils import to_networkx, degree
        dataset_obj = PygGraphPropPredDataset(name = dataset, root = 'dataset/')
        g_list = []
        class_num = 0
        for data in dataset_obj:
            # print(data.y.item())
            g_nx = to_networkx(data)
            class_num = max(class_num, data.y.item())
            g_list.append(S2VGraph(g_nx, data.y.item(), node_tags=None))
        for i, g in enumerate(g_list):
            g.edge_mat = dataset_obj[i].edge_index
            g.node_features = dataset_obj[i].x
            N = g.node_features.size(0)
            g.max_neighbor = degree(g.edge_mat[0], num_nodes=N)

            g.neighbors = [[] for i in range(N)]
            for (i, j) in g.edge_mat.T:
                g.neighbors[i].append(j)
        return g_list, class_num + 1
    elif dataset == 'ogbg-molclintox':
        from ogb.graphproppred import PygGraphPropPredDataset
        from torch_geometric.utils import to_networkx, degree
        dataset_obj = PygGraphPropPredDataset(name = dataset, root = 'dataset/')
        g_list = []
        class_num = 0
        for data in dataset_obj:
            y = torch.argmax(data.y)
            g_nx = to_networkx(data)
            class_num = max(class_num, y.item())
            g_list.append(S2VGraph(g_nx, y.item(), node_tags=None))
            
        for i, g in enumerate(g_list):
            g.edge_mat = dataset_obj[i].edge_index
            g.node_features = dataset_obj[i].x
            N = g.node_features.size(0)
            g.max_neighbor = degree(g.edge_mat[0], num_nodes=N)

            g.neighbors = [[] for i in range(N)]
            for (i, j) in g.edge_mat.T:
                g.neighbors[i].append(j)
        print(g_list[0].g)
        return g_list, class_num + 1
    else:
        with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
            n_g = int(f.readline().strip())
            for i in range(n_g):
                row = f.readline().strip().split()
                n, l = [int(w) for w in row]
                if not l in label_dict:
                    mapped = len(label_dict)
                    label_dict[l] = mapped
                g = nx.Graph()
                node_tags = []
                node_features = []
                n_edges = 0
                for j in range(n):
                    g.add_node(j)
                    row = f.readline().strip().split()
                    tmp = int(row[1]) + 2
                    if tmp == len(row):
                        # no node attributes
                        row = [int(w) for w in row]
                        attr = None
                    else:
                        row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                    if not row[0] in feat_dict:
                        mapped = len(feat_dict)
                        feat_dict[row[0]] = mapped
                    node_tags.append(feat_dict[row[0]])

                    if tmp > len(row):
                        node_features.append(attr)

                    n_edges += row[1]
                    for k in range(2, len(row)):
                        g.add_edge(j, row[k])

                if node_features != []:
                    node_features = np.stack(node_features)
                    node_feature_flag = True
                else:
                    node_features = None
                    node_feature_flag = False

                assert len(g) == n

                g_list.append(S2VGraph(g, l, node_tags))

        #add labels and edge_mat       
        for g in g_list:
            g.neighbors = [[] for i in range(len(g.g))]
            for i, j in g.g.edges():
                g.neighbors[i].append(j)
                g.neighbors[j].append(i)
            degree_list = []
            for i in range(len(g.g)):
                g.neighbors[i] = g.neighbors[i]
                degree_list.append(len(g.neighbors[i]))
            g.max_neighbor = max(degree_list)

            g.label = label_dict[g.label]

            edges = [list(pair) for pair in g.g.edges()]
            edges.extend([[i, j] for j, i in edges])

            deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
            g.edge_mat = torch.LongTensor(edges).transpose(0,1)

        if degree_as_tag:
            for g in g_list:
                g.node_tags = list(dict(g.g.degree).values())

        #Extracting unique tag labels   
        tagset = set([])
        for g in g_list:
            tagset = tagset.union(set(g.node_tags))

        tagset = list(tagset)
        tag2index = {tagset[i]:i for i in range(len(tagset))}

        for g in g_list:
            g.node_features = torch.zeros(len(g.node_tags), len(tagset))
            g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

        if verbose:
            print('# classes: %d' % len(label_dict))
            print('# maximum node tag: %d' % len(tagset))
            print("# data: %d" % len(g_list))

        return g_list, len(label_dict)

def separate_data(graph_list, seed, fold_idx):
    rs = np.random.RandomState(seed)
    rs.shuffle(graph_list)
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = False, random_state = None)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list

def getr(filepaths):
    for filepath in filepaths:
        if '.log' not in filepath:
            filepath = f"logs/{filepath}.log"
        import os
        if not os.path.exists(filepath):
            continue
        print('\n' + filepath)
        with open(filepath, 'r', encoding='utf-8') as fp:
            k,v=None,[]
            for line in fp.readlines():
                if line == '\n':
                    continue
                else:
                    import re
                    k_cur=re.findall(r'noise= (.*?),', line)[0]
                    if k is not None and k_cur != k:
                        print(f"{k} {np.mean(v)*100:.2f}({np.std(v)*100:.2f})")
                        k,v=None,[]
                    k=k_cur
                    v.append(float(re.findall(r'l_acc= (.*?),', line)[0]))
            print(f"{k} {np.mean(v)*100:.2f}({np.std(v)*100:.2f})")

def get_singular_vector(features, labels=None, n_classes=None):
    '''
    To get top1 sigular vector in class-wise manner by using SVD of hidden feature vectors
    features: hidden feature vectors of data (numpy)
    labels: correspoding label list
    '''

    if labels is not None:
        singular_vector_dict = {}
        for index in range(n_classes):
            _, _, v = np.linalg.svd(features[labels==index])
            singular_vector_dict[index] = v[0]
        return singular_vector_dict
    else:
        _, _, v = np.linalg.svd(features)
        return v[0]


