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
from sst.model import SSTModel
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


def test(model, tree_model, data_loader, args):
    def run_iter(batch, is_training, tree_model, latent_tree=True):
        model.train(is_training)
        words, length = batch.txt
        label = batch.lbl
        # tree = batch.tree
        _, tree = tree_model(words=words, length=length, tree=None)
        logits, _ = model(words=words, length=length,
                          tree=tree if not latent_tree else None)
        label_pred = logits.max(1)[1]
        accuracy = torch.eq(label, label_pred).float().mean()

        return accuracy

    test_accuracy_sum = 0
    num_test_batches = len(data_loader)
    for valid_batch in data_loader:
        test_accuracy = run_iter(
            batch=valid_batch, is_training=False,
            tree_model=tree_model,
            latent_tree=args.latent_tree)
        test_accuracy_sum += test_accuracy.item()

    test_accuracy = test_accuracy_sum / num_test_batches
    # logging.info(f'test acc {test_accuracy:.4f}')
    return test_accuracy


def load(args, num_classes, num_words):
    model = SSTModel(num_classes=num_classes, num_words=num_words,
                     word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                     clf_hidden_dim=args.clf_hidden_dim,
                     clf_num_layers=args.clf_num_layers,
                     use_leaf_rnn=args.leaf_rnn,
                     bidirectional=args.bidirectional,
                     intra_attention=args.intra_attention,
                     use_batchnorm=args.batchnorm,
                     dropout_prob=args.dropout)

    device = torch.device(args.gpu if args.gpu != -1 else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(args.tree_model))
    model.eval()

    return model


def train(args):
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
                     dropout_prob=args.dropout)
    if args.pretrained:
        model.word_embedding.weight.data.set_(TXT.vocab.vectors)
    if args.fix_word_embedding:
        logging.info('Will not update word embeddings')
        model.word_embedding.weight.requires_grad = False
    logging.info(f'Using device {args.gpu}')
    device = torch.device(args.gpu if args.gpu != -1 else 'cpu')
    model.to(device)

    tree_model = load(args, num_classes, len(TXT.vocab))

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'adam':
        optimizer_class = optim.Adam
    elif args.optimizer == 'adagrad':
        optimizer_class = optim.Adagrad
    elif args.optimizer == 'adadelta':
        optimizer_class = optim.Adadelta
    else:
        raise NotImplementedError
    optimizer = optimizer_class(params=params, weight_decay=args.l2reg)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='max', factor=0.5,
        patience=20 * args.halve_lr_every, verbose=True)
    criterion = nn.CrossEntropyLoss()

    train_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, 'log', 'train'))
    valid_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, 'log', 'valid'))

    def run_iter(batch, is_training, tree_model, latent_tree=True):
        model.train(is_training)
        words, length = batch.txt
        label = batch.lbl
        # tree = batch.tree
        _, tree = tree_model(words=words, length=length, tree=None)
        logits, _ = model(words=words, length=length,
                          tree=tree if not latent_tree else None)
        label_pred = logits.max(1)[1]
        accuracy = torch.eq(label, label_pred).float().mean()
        loss = criterion(input=logits, target=label)
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(parameters=params, max_norm=5)
            optimizer.step()
        return loss, accuracy

    def add_scalar_summary(summary_writer, name, value, step):
        if torch.is_tensor(value):
            value = value.item()
        summary_writer.add_scalar(tag=name, scalar_value=value,
                                  global_step=step)

    num_train_batches = len(train_loader)
    validate_every = num_train_batches // 20
    best_vaild_accuacy = 0
    iter_count = 0
    for epoch in range(args.max_epoch):
        for batch_iter, train_batch in enumerate(tqdm.tqdm(train_loader)):
            # for batch_iter, train_batch in enumerate(train_loader):
            train_loss, train_accuracy = run_iter(
                batch=train_batch, is_training=True,
                tree_model=tree_model,
                latent_tree=args.latent_tree)
            iter_count += 1
            add_scalar_summary(
                summary_writer=train_summary_writer,
                name='loss', value=train_loss, step=iter_count)
            add_scalar_summary(
                summary_writer=train_summary_writer,
                name='accuracy', value=train_accuracy, step=iter_count)

            if (batch_iter + 1) % validate_every == 0:
                valid_loss_sum = valid_accuracy_sum = 0
                num_valid_batches = len(valid_loader)
                for valid_batch in valid_loader:
                    valid_loss, valid_accuracy = run_iter(
                        batch=valid_batch, is_training=False,
                        tree_model=tree_model,
                        latent_tree=args.latent_tree)
                    valid_loss_sum += valid_loss.item()
                    valid_accuracy_sum += valid_accuracy.item()
                valid_loss = valid_loss_sum / num_valid_batches
                valid_accuracy = valid_accuracy_sum / num_valid_batches
                add_scalar_summary(
                    summary_writer=valid_summary_writer,
                    name='loss', value=valid_loss, step=iter_count)
                add_scalar_summary(
                    summary_writer=valid_summary_writer,
                    name='accuracy', value=valid_accuracy, step=iter_count)
                scheduler.step(valid_accuracy)
                progress = train_loader.iterations / len(train_loader)
                logging.info(f'Epoch {epoch}: '
                             f'valid loss = {valid_loss:.4f}, '
                             f'valid accuracy = {valid_accuracy:.4f}')
                if valid_accuracy > best_vaild_accuacy:
                    test_acc = test(model, tree_model, test_loader, args)
                    best_vaild_accuacy = valid_accuracy
                    model_filename = (f'nets-{progress:.2f}'
                                      f'-{valid_loss:.4f}'
                                      f'-{valid_accuracy:.4f}'
                                      f'-{test_acc:.4f}.pkl')
                    model_path = os.path.join(args.save_dir, model_filename)
                    torch.save(model.state_dict(), model_path)
                    print(f'Saved the new best nets to {model_path}')
                    logging.info(f'Test acc {test_acc:.4f}')
                if progress > args.max_epoch:
                    break


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
    parser.add_argument('--latent-tree', default=False, action='store_true')
    parser.add_argument('-seed', default=1000, type=int)
    parser.add_argument('-tree_model',
                        default='nets-0.05-1.2640-0.5144.pkl',
                        type=str)
    args = parser.parse_args()

    basic.init_seed(args.seed)
    logging.info(f'params:{args}')
    # try:
    train(args)
    # except:
    #     logging.info(f'params:{args}')


if __name__ == '__main__':
    main()
