import argparse
import os
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import psutil
import random
from tqdm import tqdm
import label_noise
from util import load_data, separate_data
from models.graphcnn import GraphCNN
from models.con import ConNN
from sklearn.mixture import GaussianMixture
import warnings
import copy
warnings.filterwarnings("ignore")

criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch, glo_state):
    model.train()

    total_iters = args.iters_per_epoch
    use_mini = False
    if total_iters == 0:
        import math
        total_iters = math.ceil(len(train_graphs) / args.batch_size)
        use_mini = True

    loss_accum = 0
    if use_mini:
        idx_rand = np.random.permutation(len(train_graphs))
    loss_all = np.empty((len(train_graphs), 1))
    for pos in range(total_iters):
        if use_mini:
            selected_idx = idx_rand[pos*args.batch_size: (pos+1)*args.batch_size]
        else:
            selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output,_ = model(batch_graph)
        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
        loss_sample = F.cross_entropy(output, labels, reduction='none')
        loss = loss_sample.mean()
        loss_all[selected_idx] = loss_sample.detach().cpu().numpy().reshape(-1, 1)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        loss = loss.detach().cpu().numpy()
        loss_accum += loss

    glo_state['prev_losses'].append(loss_all)
    average_loss = loss_accum/total_iters
    return average_loss

def train_ro(args, model, device, train_graphs, optimizer, epoch, glo_state):
    model.train()
    total_iters = args.iters_per_epoch
    use_mini = False
    if total_iters == 0:
        import math
        total_iters = math.ceil(len(train_graphs) / args.batch_size)
        use_mini = True
    
    C = np.array([graph.label for graph in train_graphs]).max() + 1

    loss_accum = 0
    if use_mini:
        idx_rand = np.random.permutation(len(train_graphs))
    loss_all = np.empty((len(train_graphs), 1))

    for pos in range(total_iters):
        if use_mini:
            selected_idx = idx_rand[pos*args.batch_size: (pos+1)*args.batch_size]
        else:
            selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]

        output, output1 = model(batch_graph)
        hidden_rep = model.hidden_rep.detach()
 
        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
        loss = F.cross_entropy(output, labels, reduction='none')
        
        loss_all[selected_idx] = loss.detach().cpu().numpy().reshape(-1, 1)

        bidx_clean = np.where(glo_state['mask_clean'][selected_idx])[0].tolist()
        bidx_noisy = np.where(~glo_state['mask_clean'][selected_idx])[0].tolist()

        l1 = torch.tensor(0.).to(device)
        l2 = torch.tensor(0.).to(device)
        if len(bidx_clean) != 0:
            clean_output = output[bidx_clean]
            y1 = labels[bidx_clean]
            l1 = F.cross_entropy(clean_output, y1, reduction='none')
            clean_labels = y1
        else:
            clean_labels = torch.LongTensor([]).to(device)
        if len(bidx_noisy) != 0:
            noisy_output = output[bidx_noisy]
            noisy_hidden_rep = hidden_rep[bidx_noisy]   
            vec_t = torch.tensor(glo_state['vec_dict']).to(device) 
            sc_in = noisy_hidden_rep @ vec_t.T 
            sc_in = F.softmax(sc_in, -1)
            sc_out1 = F.softmax(noisy_output, -1)
            sc_out2 = F.softmax(output1[bidx_noisy], -1)
            loss_weight = (epoch / args.epochs) * args.p
            sc_out = (sc_out1 + sc_out2) / 2.0
            y2_soft = (1.0 - loss_weight) * sc_in + loss_weight * sc_out
            ptu = y2_soft ** (1 / 0.5)
            y2_soft = ptu / ptu.sum(dim=1, keepdim=True)
            y2_soft = y2_soft.detach()
            hard_labels = torch.argmax(y2_soft, dim=-1)
            noisy_labels = hard_labels 
            l2 = F.mse_loss(sc_out1, y2_soft, reduction='none')
        else:
            noisy_labels = torch.LongTensor([]).to(device)
        all_labels = torch.cat([clean_labels, noisy_labels], dim=0)
        con_loss = model.con_loss1(all_labels) 
        total_loss = l1.mean() + args.loss_weight_noisy * l2.mean() + con_loss*args.cl_weight

        if optimizer is not None:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        loss_accum += total_loss.detach().cpu().numpy()

    glo_state['prev_losses'].append(loss_all)
    average_loss = loss_accum / total_iters
    return average_loss

def pass_data_iteratively(model, graphs, minibatch_size=64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        out1, _ = model([graphs[j] for j in sampled_idx])
        output.append(out1.detach())
    return torch.cat(output, 0)


def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()
    loss_fcn = nn.CrossEntropyLoss()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)

    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)

    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))
    test_loss = loss_fcn(output, labels).item()
    # probs = F.softmax(output, dim=1)
    # labels_np = labels.cpu().numpy()
    # probs_np = probs.cpu().numpy()
    # if probs_np.shape[1] > 2:
    #     rocauc = metrics.roc_auc_score(labels_np, probs_np, multi_class='ovr')
    # else:
    #     rocauc = metrics.roc_auc_score(labels_np, probs_np[:, 1])

    return acc_train, acc_test, test_loss

def calculate_and_store_losses(args, model, device, train_graphs, glo_state, batch_size=256):
    model.eval() 
    loss_all = np.empty((len(train_graphs), 1))
    with torch.no_grad():
        for start_idx in range(0, len(train_graphs), batch_size):
            end_idx = min(start_idx + batch_size, len(train_graphs))
            batch_graph = train_graphs[start_idx:end_idx]
            outputs, _ = model(batch_graph)
            labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
            loss = F.cross_entropy(outputs, labels, reduction='none')
            loss_all[start_idx:end_idx] = loss.cpu().numpy().reshape(-1, 1)
    glo_state['prev_losses'].append(loss_all)


def pass_data_iteratively_and_get_hidden_rep(model, graphs, device, minibatch_size=64):
 
    model.eval() 
    hidden_reps = []  
    idx = np.arange(len(graphs)) 
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]  
        if len(sampled_idx) == 0:
            continue
        _ = model([graphs[j] for j in sampled_idx])  
        hidden_reps.append(model.hidden_rep.detach())  
    return torch.cat(hidden_reps, 0) 

def update_model_and_state(train_graphs, model, epoch, device, args, glo_state):
    model.eval()
    C = np.array([graph.label for graph in train_graphs]).max() + 1

    if len(glo_state['prev_losses']) > 0 and (epoch == args.num_warmup + 1 or
                                              (epoch > args.num_warmup and (epoch - args.num_warmup - 1) % args.gmm_step == 0) or
                                              (glo_state['vec_dict'] is None)):
      
        loss_gmm = np.array(glo_state['prev_losses'][-5:]).mean(0)
        loss_gmm = np.array(loss_gmm).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2).fit(loss_gmm)
        gmm_prob = gmm.predict_proba(loss_gmm)  
        gmm_prob = gmm_prob[:, gmm.means_.argmin()] 
        msk_clean = gmm_prob > args.tau_1
        b1 = [train_graphs[idx] for idx in np.where(msk_clean)[0]]
        if len(b1) == 0:
            return
        y1 = torch.LongTensor([graph.label for graph in b1]).to(device)      
        all_hidden_reps = pass_data_iteratively_and_get_hidden_rep(model, train_graphs, device, minibatch_size=args.batch_size)
        clean_indices = np.where(msk_clean)[0]
        h1 = all_hidden_reps[clean_indices]
        # == Start vec_dict calculation (potentially high memory usage)
        # vec_dict = np.empty((C, h1.size(1)), dtype=np.float32)
        # for c in range(C):
        #     h1m = h1[y1 == c]
        #     from util import get_singular_vector
        #     vec = get_singular_vector(h1m.cpu().data.numpy(), None, None)  # (D,)
        #     vec_dict[c] = vec
        vec_dict = np.empty((C, h1.size(1)), dtype=np.float32)
        for c in range(C):
            h1m = h1[y1 == c] 
            if len(h1m) > 0:
                gram_matrix = torch.zeros((h1m.shape[1], h1m.shape[1]), dtype=torch.float32,device=device)
                for z_i in h1m:
                    z_i = z_i.view(-1, 1)
                    gram_matrix += z_i @ z_i.t()
                gram_matrix_cpu = gram_matrix.cpu().numpy()
                from util import get_singular_vector
                vec = get_singular_vector(gram_matrix_cpu, None, None)
                vec_dict[c] = vec
            else:
                vec_dict[c] = np.zeros((h1.size(1),), dtype=np.float32)
      
        train_idx_clean = []
        wloss = np.empty((len(train_graphs), 2), dtype=np.float32)
 
        hidden_rep_ = all_hidden_reps
        labels_ = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
        for c in range(C):
            idx_class = torch.where(labels_ == c)[0].cpu().data.numpy()
            score = hidden_rep_ @ torch.tensor(vec_dict[c]).to(device).unsqueeze(-1)
            ss = score[labels_ == c]
            ss = ss.cpu().data.numpy() 
            gmm = GaussianMixture(n_components=2).fit(ss)
            s_prob = gmm.predict_proba(ss) 
            wloss[idx_class] = s_prob.copy()
            s_prob = s_prob[:, gmm.means_.argmax()] 

            msk_s = s_prob > args.tau_2
            train_idx_clean += torch.where(labels_ == c)[0][msk_s].cpu().data.tolist()

        glo_state['mask_clean'] = np.zeros(len(train_graphs), dtype=np.bool_)
        glo_state['mask_clean'][train_idx_clean] = True
        glo_state['vec_dict'] = vec_dict.copy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default='MUTAG',
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=0,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=16,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    parser.add_argument('--ptb', type=float, default=0.3, 
                    help="noise ptb_rate")
    parser.add_argument('--noise', type=str, default='uniform', choices=['uniform', 'pair'], 
                    help='type of noises')
    parser.add_argument('--variant', type=int, default=2, help='[all, w/o score]')
    parser.add_argument('--num_warmup', type=int, default=5)
    parser.add_argument('--tau_1', type=float, default=0.7)
    parser.add_argument('--tau_2', type=float, default=0.5)
    parser.add_argument('--loss_weight', type=float, default=0.5)
    parser.add_argument('--loss_weight_noisy', type=float, default=0.5)
    parser.add_argument('--cl_weight', type=float, default=1.0)
    parser.add_argument('--gmm_step', type=int, default=5)
    parser.add_argument('--p', type=float, default=0.6)

    parser.add_argument('--use_gce', type=int, default=False)
    parser.add_argument('--lam_gce', type=float, default=0.5)
    parser.add_argument('--q_gce', type=float, default=0.7)

    parser.add_argument('--con_eta', type=float, default=1.0)
    parser.add_argument('--con_lam', type=float, default=1.0)

    parser.add_argument('--k_list', type=int, nargs='+', default=[1,5,10,20])

    args = parser.parse_args()

    if args.dataset in ['MUTAG','PTC','COX2','ENZYMES']:
        args.hidden_dim=16
        args.batch_size=32
    elif args.dataset in ['NCI1', 'PROTEINS','DD']:
        args.hidden_dim=32
        args.batch_size=128
    else :
        args.hidden_dim=64
        args.batch_size=128

    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    accuracies = [[] for i in range(args.epochs)] 
    graphs, num_classes = load_data(args.dataset, args.degree_as_tag)
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

    ptb = args.ptb
    noise_type = args.noise
    train_labels = [graph.label  for graph in train_graphs]
    train_labels = np.array(train_labels)
    noise_y, P = label_noise.noisify_p(train_labels, num_classes, ptb, noise_type, args.seed)
    for i in range(len(train_graphs)):
        train_graphs[i].label = noise_y[i]
    
    enc1 = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)
    import copy
    enc2 = copy.deepcopy(enc1)
    model = ConNN(args.hidden_dim*(args.num_layers-1), args.hidden_dim*(args.num_layers-1), enc1, enc2, eta=args.con_eta, lam=args.con_lam).to(device)
    optimizer = optim.Adam(model.get_used_parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    best_acc = 0.0
    glo_state = {
        'mask_clean': np.zeros(len(train_graphs), dtype=np.bool_),
        'prev_losses': [],
        'vec_dict': None,
    }
    import time
    stime = time.time()
    for epoch in range(1, args.epochs + 1):
        # scheduler.step()
        if epoch <= args.num_warmup:
            avg_loss = train(args, model, device, train_graphs, optimizer, epoch, glo_state)
        else:
            update_model_and_state(train_graphs, model,epoch, device, args, glo_state)
            avg_loss = train_ro(args, model, device, train_graphs, optimizer, epoch, glo_state)         
        acc_train, acc_test, test_loss = test(args, model, device, train_graphs, test_graphs, epoch)
        train_accuracies.append(acc_train)
        test_accuracies.append(acc_test)
        test_losses.append(test_loss)
        accuracies[epoch-1].append(acc_test)
        print(f"Epoch {epoch}/{args.epochs} - Train acc: {acc_train*100:.2f}%, Test acc: {acc_test*100:.2f}%")
        if not args.filename == "":
            with open(args.filename, 'w') as f:
                f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                f.write("\n")
        if acc_test> best_acc:
            best_acc = acc_test
    k_list = args.k_list
    last_acc_k = np.array([(test_accuracies[-k]) for k in k_list]) * 100
    best_iter_acc = test_accuracies[np.argmax(train_accuracies)]
    np.set_printoptions(precision=2)
    print(f'Best acc: {best_acc*100:.2f}, last acc {k_list}: {last_acc_k},',
        f'best_iter_acc= {best_iter_acc*100:.2f}, Time: {time.time()-stime}')
    

