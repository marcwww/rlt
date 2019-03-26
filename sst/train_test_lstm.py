import argparse
import logging
import os
import tqdm

from tensorboardX import SummaryWriter

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torchtext import data, datasets
from torchtext.data import Dataset
import numpy as np
from nets import basic
import six
from nltk.tree import Tree
from sst.model_lstm import SSTModel
import crash_on_ipy

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

class Example(object):

    def __init__(self, txt, tree, lbl):
        self.txt = txt
        self.tree = tree
        self.lbl = lbl

def brackets2indices(brackets):
    def new_brackets(brackets, bracket_chosen):
        res = set()
        base = bracket_chosen[1]
        for bracket in brackets:
            if bracket != bracket_chosen:
                l, r = bracket
                l = l - 1 if l >= base else l
                r = r - 1 if r >= base else r
                res.add((l, r))
        return res

    indices = []
    while len(brackets) > 1:
        bracket_chosen = None
        for bracket in brackets:
            if bracket[1] - bracket[0] == 2:
                bracket_chosen = bracket
                break
        # print(sorted(brackets))
        # print(bracket_chosen)
        brackets = new_brackets(brackets, bracket_chosen)
        # print(sorted(brackets))
        # print('-' * 50)
        indices.append(bracket_chosen[0])
    return indices

def load_examples(fname, subtrees=False):
    examples = []
    lens = []
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            tree = Tree.fromstring(line.strip())
            if subtrees:
                for t in tree.subtrees():
                    txt = t.leaves()
                    lens.append(len(txt))
                    lbl = t.label()
                    brackets = basic.get_brackets(t)[0]
                    brackets.add((0, len(txt)))
                    brackets = sorted(brackets)
                    indices = brackets2indices(brackets)
                    assert len(indices) + 2 == len(txt) or len(txt) == 1
                    examples.append(Example(txt, indices, lbl))
            else:
                t = tree
                txt = t.leaves()
                lens.append(len(txt))
                lbl = t.label()
                brackets = basic.get_brackets(t)[0]
                brackets.add((0, len(txt)))
                brackets = sorted(brackets)
                indices = brackets2indices(brackets)
                assert len(indices) + 2 == len(txt) or len(txt) == 1
                examples.append(Example(txt, indices, lbl))
    return examples, np.mean(lens)

def build_iters(args):
    TXT = data.Field(lower=args.lower,
                     include_lengths=True,
                     batch_first=True)
    LBL = data.Field(sequential=False,
                     unk_token=None)
    TREE = data.Field(sequential=True,
                      use_vocab=False,
                      pad_token=0)

    ftrain = 'data/sst/sst/trees/train.txt'
    fvalid = 'data/sst/sst/trees/dev.txt'
    ftest = 'data/sst/sst/trees/test.txt'

    examples_train, len_ave = load_examples(ftrain, subtrees=True)
    examples_valid, _ = load_examples(fvalid, subtrees=False)
    examples_test, _ = load_examples(ftest, subtrees=False)
    train = Dataset(examples_train, fields=[('txt', TXT),
                                    ('tree', TREE),
                                    ('lbl', LBL)])
    TXT.build_vocab(train, vectors=args.pretrained)
    LBL.build_vocab(train)
    valid = Dataset(examples_valid, fields=[('txt', TXT),
                                    ('tree', TREE),
                                    ('lbl', LBL)])
    test = Dataset(examples_test, fields=[('txt', TXT),
                                          ('tree', TREE),
                                          ('lbl', LBL)])

    def batch_size_fn(new_example, current_count, ebsz):
        return ebsz + (len(new_example.txt) / len_ave) ** 0.3

    device = torch.device(args.gpu if args.gpu != -1 else 'cpu')
    train_iter = basic.BucketIterator(train,
                                      batch_size=args.batch_size,
                                      sort=True,
                                      shuffle=True,
                                      repeat=False,
                                      sort_key=lambda x: len(x.txt),
                                      batch_size_fn=batch_size_fn,
                                      device=device)

    valid_iter = basic.BucketIterator(valid,
                                      batch_size=args.batch_size,
                                      sort=True,
                                      shuffle=True,
                                      repeat=False,
                                      sort_key=lambda x: len(x.txt),
                                      batch_size_fn=batch_size_fn,
                                      device=device)

    test_iter = basic.BucketIterator(test,
                                     batch_size=args.batch_size,
                                     sort=True,
                                     shuffle=True,
                                     repeat=False,
                                     sort_key=lambda x: len(x.txt),
                                     batch_size_fn=batch_size_fn,
                                     device=device)

    return train_iter, valid_iter, test_iter, (TXT, TREE, LBL)

def load_and_test(args):

    train_loader, valid_loader, test_loader, (TXT, TREE, LBL) = build_iters(args)

    num_classes = len(LBL.vocab)
    model = SSTModel(num_classes=num_classes, num_words=len(TXT.vocab),
                     word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                     clf_hidden_dim=args.clf_hidden_dim,
                     clf_num_layers=args.clf_num_layers,
                     use_leaf_rnn=args.leaf_rnn,
                     bidirectional=args.bidirectional,
                     intra_attention=args.intra_attention,
                     use_batchnorm=args.batchnorm,
                     dropout_prob=args.dropout,
                     pooling=args.pooling)
    if args.pretrained:
        model.word_embedding.weight.data.set_(TXT.vocab.vectors)
    if args.fix_word_embedding:
        logging.info('Will not update word embeddings')
        model.word_embedding.weight.requires_grad = False
    logging.info(f'Using device {args.gpu}')
    device = torch.device(args.gpu if args.gpu != -1 else 'cpu')
    model.to(device)

    if args.load is not None:
        model.load_state_dict(torch.load(args.load))
        logging.info(f'loaded from {args.load}')

    def run_iter(batch, is_training, latent_tree=True):
        model.train(is_training)
        words, length = batch.txt
        label = batch.lbl
        tree = batch.tree
        logits = model(words=words, length=length,
                       tree=tree if not latent_tree else None)
        label_pred = logits.max(1)[1]
        accuracy = torch.eq(label, label_pred).float().mean()
        return accuracy

    data_loader = test_loader

    test_accuracy_sum = 0
    num_test_batches = len(data_loader)
    for valid_batch in data_loader:
        test_accuracy = run_iter(
            batch=valid_batch, is_training=False,
            latent_tree=args.latent_tree)
        test_accuracy_sum += test_accuracy.item()

    test_accuracy = test_accuracy_sum / num_test_batches
    logging.info(f'test acc {test_accuracy:.4f}')
    return test_accuracy

def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--word-dim', type=int, default=300)
    parser.add_argument('--hidden-dim', type=int, default=300)
    parser.add_argument('--clf-hidden-dim', type=int, default=1024)
    parser.add_argument('--clf-num-layers', type=int, default=1)
    parser.add_argument('--leaf-rnn', default=True, action='store_true')
    # parser.add_argument('--leaf-rnn', default=False, action='store_true')
    # parser.add_argument('--bidirectional', default=True, action='store_true')
    parser.add_argument('--bidirectional', default=False, action='store_true')
    parser.add_argument('--intra-attention', default=False, action='store_true')
    parser.add_argument('--batchnorm', default=False, action='store_true')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--l2reg', default=0.0, type=float)
    parser.add_argument('--pretrained', default='glove.840B.300d')
    # parser.add_argument('--pretrained', default=None)
    parser.add_argument('--fix-word-embedding', default=False,
                        action='store_true')
    parser.add_argument('-gpu', default=-1, type=int)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-epoch', type=int, default=1000)
    parser.add_argument('--save-dir', type=str, default='.')
    parser.add_argument('--omit-prob', default=0.0, type=float)
    parser.add_argument('--optimizer', default='adadelta')
    parser.add_argument('--fine-grained', default=True, action='store_true')
    parser.add_argument('--halve-lr-every', default=2, type=int)
    parser.add_argument('--lower', default=False, action='store_true')
    # parser.add_argument('--latent-tree', default=True, action='store_true')
    parser.add_argument('--latent-tree', default=False, action='store_true')
    parser.add_argument('-seed', default=1000, type=int)
    parser.add_argument('-load', default='nets-0.30-1.2069-0.4980.pkl', type=str)
    parser.add_argument('-pooling', default='lstm', type=str)

    args = parser.parse_args()

    basic.init_seed(args.seed)
    logging.info(f'params:{args}')
    load_and_test(args)


if __name__ == '__main__':
    main()