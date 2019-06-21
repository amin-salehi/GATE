import argparse

from utils.classifier import Classifier
from trainer import Trainer
from utils import process


def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Run gate.")

    parser.add_argument('--dataset', nargs='?', default='cora',
                        help='Input dataset')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate. Default is 0.001.')

    parser.add_argument('--n-epochs', default=200, type=int,
                        help='Number of epochs')

    parser.add_argument('--hidden-dims', type=list, nargs='+', default=[512, 512],
                        help='Number of dimensions.')

    parser.add_argument('--lambda-', default=1, type=float,
                        help='Parameter controlling the contribution of edge reconstruction in the loss function.')

    parser.add_argument('--dropout', default=0.0, type=float,
                        help='Dropout.')

    parser.add_argument('--gradient_clipping', default=5.0, type=float,
                        help='gradient clipping')

    return parser.parse_args()

def main(args):
    '''
    Pipeline for Graph Attention Autoencoder.
    '''

    G, X, Y, idx_train, idx_val, idx_test = process.load_data(args.dataset)

    # add feature dimension size to the beginning of hidden_dims
    feature_dim = X.shape[1]
    args.hidden_dims = [feature_dim] + args.hidden_dims

    # prepare the data
    G_tf,  S, R = process.prepare_graph_data(G)

    # Train the Model
    trainer = Trainer(args)
    trainer(G_tf, X, S, R)
    embeddings, attentions = trainer.infer(G_tf, X, S, R)

    # Evaluate the quality of embeddings
    classifier = Classifier(vectors=embeddings)
    f1s = classifier(idx_train, idx_test, idx_val, Y, seed=0)
    print f1s

if __name__ == "__main__":
    args = parse_args()
    main(args)