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
import random

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

class ReplayMem(object):

    def __init__(self, nbatch):
        self.mem = [None for _ in range(nbatch)]
        self.ptr = 0

    def sample(self):
        return random.choice(self.mem)

    def push(self, batch):
        self.mem[self.ptr % len(self.mem)] = batch
        self.ptr += 1

def load_mem(mem: ReplayMem, model, data_iter):
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(data_iter)):
            expr, length = batch.expr
            val = batch.val
            clf_logits, log_prob_sum_old, tree, _ = model(inp=expr, length=length)
            clf_logits_baseline, _, _, _ = model(inp=expr, length=length, self_critic=True)
            reward = (-F.cross_entropy(clf_logits, val, reduce=False)).exp()
            reward_baseline = (-F.cross_entropy(clf_logits_baseline, val, reduce=False)).exp()
            advan = normalize_reward(reward - reward_baseline)
            mem.push((expr, length, val, tree, advan, log_prob_sum_old))

def normalize_reward(r):
    return r.div(r.std())

def run_iter(model, optimizer, params, batch, mem: ReplayMem, is_training, train_parser):
    model.train(is_training)

    if train_parser:
        model.train_parser()
        expr, length, val, tree, advan, log_prob_sum_old = mem.sample()
        clf_logits, log_prob_sum_new, tree_new, entropy = model(inp=expr, length=length, tree=tree)
        r = (log_prob_sum_new - log_prob_sum_old).exp()
        r_clipped = torch.clamp(r, min=1-0.25, max=1+0.25)
        advan_mask = advan.gt(0).float()
        effect_bsz = max(advan_mask.sum(), 1)
        # advan.masked_fill_(advan_mask, 0)
        advan = advan * advan_mask
        ppo_loss = -torch.min(r * advan, r_clipped * advan).sum().div(effect_bsz)
        loss = ppo_loss - 1e-2 * entropy.sum().div(effect_bsz)
        # ppo_loss = -torch.min(r * advan, r_clipped * advan).mean()
        # loss = ppo_loss - 1e-2 * entropy.mean()
        with torch.no_grad():
            expr, length = batch.expr
            val = batch.val
            clf_logits, log_prob_sum_old, tree, _ = model(inp=expr, length=length)
            clf_logits_baseline, _, _, _ = model(inp=expr, length=length, self_critic=True)
            reward = (-F.cross_entropy(clf_logits, val, reduce=False)).exp()
            reward_baseline = (-F.cross_entropy(clf_logits_baseline, val, reduce=False)).exp()
            advan = normalize_reward(reward - reward_baseline)
            mem.push((expr, length, val, tree, advan, log_prob_sum_old))
    else:
        model.train_composition()
        expr, length = batch.expr
        val = batch.val
        clf_logits, _, _, _ = model(inp=expr, length=length, self_critic=True)
        loss = F.cross_entropy(clf_logits, val)

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

    logging.info(f'Using device {args.gpu}')
    device = torch.device(args.gpu if args.gpu != -1 else 'cpu')
    model.to(device)

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

    mem = ReplayMem(nbatch=len(train_iter))
    logging.info('Loading mem ...')
    load_mem(mem, model, data_iter=train_iter)
    logging.info('Loading mem done.')

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
                                     optimizer,
                                     params,
                                     batch,
                                     mem,
                                     is_training=True,
                                     train_parser=False)
                loss_comp.append(loss.item())
            else:
                # train parser
                loss, acc = run_iter(model,
                                     optimizer,
                                     params,
                                     batch,
                                     mem,
                                     is_training=True,
                                     train_parser=True)
                loss_parser.append(loss.item())

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
                             optimizer,
                             params,
                             batch_valid,
                             mem,
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
