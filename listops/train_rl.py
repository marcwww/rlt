import argparse
import logging
import os

from tensorboardX import SummaryWriter

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from torchtext import data, datasets
import crash_on_ipy
import torchtext
from torchtext.data import Dataset
import numpy as np
import re

from listops.macros import *
from nets import basic
from listops.model_rl import ListopsModel
import tqdm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def tokenize(expr):
    return re.findall(r'[^ \(\)]+', expr)


class Example(object):
    def __init__(self, expr, val):
        self.expr = expr
        self.val = int(val)


def load_examples(fname, seq_len_max=None):
    examples = []
    ndiscard = 0
    lens = []

    with open(fname, 'r') as f:
        for line in f:
            val, expr = line.strip().split('\t')
            expr = tokenize(expr)
            if seq_len_max == None or len(expr) <= seq_len_max:
                examples.append(Example(expr, val))
                lens.append(len(expr))
            else:
                ndiscard += 1
        logging.info(f'Discarded {ndiscard} examples.')

    len_ave = np.mean(lens)
    return examples, len_ave


def build_iters(args):
    EXPR = torchtext.data.Field(sequential=True,
                                use_vocab=True,
                                batch_first=True,
                                include_lengths=True,
                                pad_token=PAD,
                                eos_token=None)
    VAL = torchtext.data.Field(sequential=False,
                               unk_token=None)
    ftrain = 'data/train_d20s.tsv'
    fvalid = 'data/test_d20s.tsv'
    # ftest = 'data/test_d20s.tsv'

    examples_train, len_ave = load_examples(ftrain, seq_len_max=100)
    # examples_train, len_ave = load_examples(ftrain)
    examples_valid, _ = load_examples(fvalid)
    train = Dataset(examples_train, fields=[('expr', EXPR),
                                            ('val', VAL)])
    EXPR.build_vocab(train)
    VAL.build_vocab(train)
    valid = Dataset(examples_valid, fields=[('expr', EXPR),
                                            ('val', VAL)])

    def batch_size_fn(new_example, current_count, ebsz):
        return ebsz + (len(new_example.expr) / len_ave) ** 0.3
        # return ebsz + (len(new_example.expr) / len_ave) ** 1

    device = torch.device(args.gpu if args.gpu != -1 else 'cpu')
    train_iter = basic.BucketIterator(train,
                                      batch_size=args.bsz,
                                      sort=True,
                                      shuffle=True,
                                      repeat=False,
                                      sort_key=lambda x: len(x.expr),
                                      batch_size_fn=batch_size_fn,
                                      device=device)

    valid_iter = basic.BucketIterator(valid,
                                      batch_size=args.bsz,
                                      sort=True,
                                      shuffle=True,
                                      repeat=False,
                                      sort_key=lambda x: len(x.expr),
                                      batch_size_fn=batch_size_fn,
                                      device=device)

    return train_iter, valid_iter, EXPR, VAL


def run_iter(model, model_old, criterion, optimizer,
             params, batch, is_training, train_parser):
    model.train(is_training)
    expr, length = batch.expr
    val = batch.val
    if train_parser:
        model.train_parser()
        with torch.no_grad():
            clf_logits, log_prob_sum_old, tree, _ = model_old(inp=expr, length=length)
            clf_logits_baseline, _, _, _ = model(inp=expr, length=length, self_critic=True)
        _, log_prob_sum, _, entropy = model(inp=expr, length=length, tree=tree)

        A = F.cross_entropy(clf_logits, val, reduce=False) - \
            F.cross_entropy(clf_logits_baseline, val, reduce=False)
        # A = F.cross_entropy(clf_logits, val, reduce=False)
        # A_normalized = (A - A.mean()).div(A.std())
        A_normalized = A.div(A.std())
        # A_normalized = A
        r = (log_prob_sum - log_prob_sum_old).exp()
        r_clipped = torch.clamp(r, min=1 - 0.25, max=1 + 0.25)
        surrogate_loss = (r * A_normalized).mean()
        surrogate_loss_clipped = (r_clipped * A_normalized).mean()
        loss = max(surrogate_loss, surrogate_loss_clipped) - 1e-4 * entropy.mean()
    else:
        model.train_composition()
        clf_logits, _, _, _ = model(inp=expr, length=length, self_critic=True)
        loss = criterion(clf_logits, val)

    if is_training:
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(parameters=params, max_norm=5)
        optimizer.step()

    pred = clf_logits.max(dim=1)[1]
    acc = torch.eq(val, pred).float().mean()

    return loss, acc


def train(args):
    train_iter, valid_iter, EXPR, VAL = build_iters(args)
    model = ListopsModel(nclasses=len(VAL.vocab),
                         nwords=len(EXPR.vocab),
                         edim=args.edim,
                         hdim=args.hdim)
    model_old = ListopsModel(nclasses=len(VAL.vocab),
                             nwords=len(EXPR.vocab),
                             edim=args.edim,
                             hdim=args.hdim)
    model_old.load_state_dict(model.state_dict())

    logging.info(f'Using device {args.gpu}')
    device = torch.device(args.gpu if args.gpu != -1 else 'cpu')
    model.to(device)
    model_old.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'adam':
        optim_class = optim.Adam
    elif args.optimizer == 'adagrad':
        optim_class = optim.Adagrad
    elif args.optimizer == 'adadelta':
        optim_class = optim.Adadelta
    else:
        raise NotImplementedError
    optimizer = optim_class(params=params, weight_decay=args.l2reg)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                               mode='max',
                                               factor=0.5,
                                               patience=20 * args.halve_lr_every,
                                               verbose=True)
    criterion = nn.CrossEntropyLoss()
    num_train_batches = len(train_iter)
    valid_every = num_train_batches // 1
    best_valid_acc = 0

    for epoch in range(args.nepoches):
        train_iter_tqdm = tqdm.tqdm(train_iter)
        loss_parser = []
        loss_comp = []
        for i, batch in enumerate(train_iter_tqdm):
            if i % (args.parser_batch + 1) == 0:
                # train composition function
                loss, acc = run_iter(model,
                                     model_old,
                                     criterion,
                                     optimizer,
                                     params,
                                     batch,
                                     is_training=True,
                                     train_parser=False)
                loss_comp.append(loss.item())
            else:
                # train parser
                loss, acc = run_iter(model,
                                     model_old,
                                     criterion,
                                     optimizer,
                                     params,
                                     batch,
                                     is_training=True,
                                     train_parser=True)
                loss_parser.append(loss.item())
            if i % (10 + 1) == 0:
                model_old.load_state_dict(model.state_dict())

        print(f'Epoch={epoch} '
              f'Loss(parser)={np.mean(loss_parser):.4f} '
              f'Loss(comp)={np.mean(loss_comp):.4f}')

        # valid
        with torch.no_grad():
            valid_losses = []
            valid_accs = []
            for batch_valid in tqdm.tqdm(valid_iter):
                valid_loss, valid_acc = \
                    run_iter(model,
                             model_old,
                             criterion,
                             optimizer,
                             params,
                             batch_valid,
                             is_training=False,
                             train_parser=False)
                valid_losses.append(valid_loss.item())
                valid_accs.append(valid_acc.item())
            valid_acc = np.mean(valid_accs)
            valid_loss = np.mean(valid_losses)
            scheduler.step(valid_acc)
            logging.info(f'Epoch {epoch} '
                         f'Valid loss {valid_loss:.4f} '
                         f'Valid acc {valid_acc:.4f} ')
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                model_fname = (f'epoch-{epoch}'
                               f'loss-{valid_loss:.4f}'
                               f'acc-{valid_acc:4f}.pkl')
                torch.save(model.state_dict(), model_fname)
                logging.info(f'Saving to {model_fname}')


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('-edim', type=int, default=128)
    parser.add_argument('-hdim', type=int, default=128)
    # parser.add_argument('-bsz', type=int, default=32)
    parser.add_argument('-bsz', type=int, default=128)
    parser.add_argument('-l2reg', type=float, default=0.0)
    parser.add_argument('-gpu', type=int, default=-1)
    parser.add_argument('-optimizer', type=str, default='adadelta')
    parser.add_argument('-nepoches', type=int, default=1000)
    parser.add_argument('-parser_batch', type=int, default=15)
    parser.add_argument('--halve-lr-every', type=int, default=2)
    parser.add_argument('-seed', type=int, default=1000)

    args = parser.parse_args()
    basic.init_seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()