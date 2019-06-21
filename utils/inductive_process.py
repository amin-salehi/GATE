import cPickle
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys

from scipy import sparse


def prepare_graph_data(adj):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    data =  adj.tocoo().data
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape), adj.row, adj.col
    #return (indices, adj.data, adj.shape), adj.row, adj.col, data#adj.data

def prepare_graph_data1(adj):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    data =  adj.tocoo().data
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape), adj.row, adj.col
    #return (indices, adj.data, adj.shape), adj.row, adj.col, data#adj.data

def conver_sparse_tf2np(input):
    # Convert Tensorflow sparse matrix to Numpy sparse matrix
    return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])), shape=(input[layer][2][0], input[layer][2][1])) for layer in input]

###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    nx_graph = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(nx_graph)


    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    #
    # print(adj.shape)
    # print(features.shape)


    ################# added section to extract train subgraph ######################
    ids = set(range(labels.shape[0]))
    train_ids = ids.difference(set(idx_val + idx_test))
    # train_edges = [edge for edge in nx_graph.edges() if edge[0] in train_ids and edge[1] in train_ids]
    #
    # adj_train = sparse.dok_matrix((len(ids), len(ids)))
    # for edge in train_edges:
    #     if edge[0] != edge[1]:
    #         adj_train[edge[0], edge[1]] = 1

    nx_train_graph = nx_graph.subgraph(train_ids)
    adj_train = nx.adjacency_matrix(nx_train_graph)

    features = features.todense()
    features_train = features[np.array(list((train_ids)))]

    ################################################################################


    return adj_train, adj, features_train, features, labels, idx_train, idx_val, idx_test


def load_nell_data(DATASET='nell'):
    NAMES = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    OBJECTS = []
    for i in range(len(NAMES)):
        OBJECTS.append(cPickle.load(open('data/ind.{}.{}'.format(DATASET, NAMES[i]), 'rb')))
    x, y, tx, ty, allx, ally, graph = tuple(OBJECTS)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(DATASET))
    exclu_rang = []
    for i in range(8922, 65755):
        if i not in test_idx_reorder:
            exclu_rang.append(i)

    # get the features:X
    allx_v_tx = sp.vstack((allx, tx)).tolil()
    _x = sp.lil_matrix(np.zeros((9891, 55864)))

    up_features = sp.hstack((allx_v_tx, _x))

    _x = sp.lil_matrix(np.zeros((55864, 5414)))
    _y = sp.identity(55864, format='lil')
    down_features = sp.hstack((_x, _y))
    features = sp.vstack((up_features, down_features)).tolil()
    features[test_idx_reorder + exclu_rang, :] = features[range(8922, 65755), :]
    print "Feature matrix:" + str(features.shape)

    # get the labels: y
    up_labels = np.vstack((ally, ty))
    down_labels = np.zeros((55864, 210))
    labels = np.vstack((up_labels, down_labels))
    labels[test_idx_reorder + exclu_rang, :] = labels[range(8922, 65755), :]
    print "Label matrix:" + str(labels.shape)

    # print np.sort(graph.get(17493))

    # get the adjcent matrix: A
    # adj = nx.to_numpy_matrix(nx.from_dict_of_lists(graph))
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    print "Adjcent matrix:" + str(adj.shape)

    # test, validation, train
    idx_test = test_idx_reorder
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]



    ################# added section to extract train subgraph ######################
    nx_train_graph = G.subgraph(idx_train + idx_val)
    adj_train = nx.adjacency_matrix(nx_train_graph)
    features = features.todense()
    train_features = features[np.array(idx_train + idx_val)]
    ################################################################################

    # # record the intermedia result for saving time
    # cPickle.dump(adj, open('data/cell.adj.pkl', 'wb'))
    # cPickle.dump(features, open('data/cell.features.pkl', 'wb'))
    # cPickle.dump(y_train, open('data/cell.yTrain.pkl', 'wb'))
    # cPickle.dump(y_val, open('data/cell.yVal.pkl', 'wb'))
    # cPickle.dump(y_test, open('data/cell.yTest.pkl', 'wb'))
    # cPickle.dump(train_mask, open('data/cell.trainMask.pkl', 'wb'))
    # cPickle.dump(val_mask, open('data/cell.valMask.pkl', 'wb'))
    # cPickle.dump(test_mask, open('data/cell.testMask.pkl', 'wb'))

   # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    return adj_train, adj, train_features, features, labels, idx_train, idx_val, idx_test


def load_random_data(size):

    adj = sp.random(size, size, density=0.002) # density similar to cora
    features = sp.random(size, 1000, density=0.015)
    int_labels = np.random.randint(7, size=(size))
    labels = np.zeros((size, 7)) # Nx7
    labels[np.arange(size), int_labels] = 1

    train_mask = np.zeros((size,)).astype(bool)
    train_mask[np.arange(size)[0:int(size/2)]] = 1

    val_mask = np.zeros((size,)).astype(bool)
    val_mask[np.arange(size)[int(size/2):]] = 1

    test_mask = np.zeros((size,)).astype(bool)
    test_mask[np.arange(size)[int(size/2):]] = 1

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
  
    # sparse NxN, sparse NxF, norm NxC, ..., norm Nx1, ...
    #return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    return

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
    # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
    return indices, adj.data, adj.shape

