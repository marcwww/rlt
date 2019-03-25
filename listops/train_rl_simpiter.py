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

    examples_train, len_ave = load_examples(ftrain, seq_len_max=args.max_length)
    # examples_train, len_ave = load_examples(ftrain)
    examples_valid, _ = load_examples(fvalid)
    train = Dataset(examples_train, fields=[('expr', EXPR),
                                            ('val', VAL)])
    EXPR.build_vocab(train)
    VAL.build_vocab(train)
    valid = Dataset(examples_valid, fields=[('expr', EXPR),
                                            ('val', VAL)])

    device = torch.device(args.gpu if args.gpu != -1 else 'cpu')
    train_iter = torchtext.data.Iterator(train,
                                         batch_size=args.bsz,
                                         shuffle=True,
                                         repeat=False,
                                         device=device)
    valid_iter = torchtext.data.Iterator(valid,
                                         batch_size=args.bsz,
                                         shuffle=True,
                                         repeat=False,
                                         device=device)

    return train_iter, valid_iter, EXPR, VAL

class ReplayMem(object):

    def __init__(self):
        self.mem = []

    def push(self, batch):
        self.mem.append(batch)

    def __len__(self):
        return len(self.mem)

    def clear(self):
        self.mem.clear()

def xent(logits, tar):
    probs = F.softmax(logits, dim=1)
    log_inv_prob = torch.log(1 - probs)
    reward = -torch.gather(log_inv_prob,
                               dim=1,
                               index=tar.view(-1, 1))
    return reward.squeeze()

def lllhood(logits, tar):
    probs = F.softmax(logits, dim=1)
    log_inv_prob = torch.log(probs)
    reward = torch.gather(log_inv_prob,
                           dim=1,
                           index=tar.view(-1, 1))
    return reward.squeeze()

def load_mem(mem: ReplayMem, model, batch):
    with torch.no_grad():
        expr, length = batch.expr
        val = batch.val
        clf_logits, log_prob_sum_old, tree, _ = model(inp=expr, length=length)
        clf_logits_baseline, _, _, _ = model(inp=expr, length=length, self_critic=True)
        # reward = xent(clf_logits, val)
        # reward_baseline = xent(clf_logits_baseline, val)
        reward = lllhood(clf_logits, val)
        reward_baseline = lllhood(clf_logits_baseline, val)
        advan = normalize(reward - reward_baseline)
        mem.push((expr, length, val, tree, advan, log_prob_sum_old))

# def normalize(r):
#     return (r - r.mean()).div(r.std() + 1e-8)
def normalize(r):
    return r.div(r.std() + 1e-8)

def run_iter(model, optimizer, params, batch, is_training, train_parser, entropy_coef=1e-2):
    model.train(is_training)

    if train_parser:
        model.train_parser()
        expr, length, val, tree, advan, log_prob_sum_old = batch
        clf_logits, log_prob_sum_new, tree_new, entropy = model(inp=expr, length=length, tree=tree)
        r = (log_prob_sum_new - log_prob_sum_old).exp()
        r_clipped = torch.clamp(r, min=1-0.25, max=1+0.25)
        loss_policy = -torch.min(r * advan, r_clipped * advan).mean()
        loss_entropy = entropy.mean()
        loss = loss_policy - entropy_coef * loss_entropy
    else:
        model.train_composition()
        expr, length = batch.expr
        val = batch.val
        clf_logits, _, _, _ = model(inp=expr, length=length, self_critic=True)
        loss = F.cross_entropy(clf_logits, val)
        loss_entropy = None

    if is_training:
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(parameters=params, max_norm=5)
        optimizer.step()

    pred = clf_logits.max(dim=1)[1]
    acc = torch.eq(val, pred).float().mean()

    return loss, acc, loss_entropy


def train(args):
    train_iter, valid_iter, EXPR, VAL = build_iters(args)
    model = ListopsModel(nclasses=len(VAL.vocab),
                         nwords=len(EXPR.vocab),
                         edim=args.edim,
                         hdim=args.hdim)

    logging.info(f'Using device {args.gpu}')
    device = torch.device(args.gpu if args.gpu != -1 else 'cpu')
    model.to(device)

    if args.load != None:
        model.load_state_dict(torch.load(args.load))
        logging.info(f'Loading from {args.load}')

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
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
    #                                            mode='max',
    #                                            factor=0.5,
    #                                            patience=20 * args.halve_lr_every,
    #                                            verbose=True)

    mem = ReplayMem()
    best_valid_acc = 0

    for epoch in range(args.nepoches):
        model.train()
        train_iter_tqdm = tqdm.tqdm(train_iter, ncols=100)
        loss_parser = []
        loss_comp = []
        entropy_lst = []
        for i, batch in enumerate(train_iter_tqdm):
            if len(mem) != args.K:
                # loading examples to replay memory
                load_mem(mem, model, batch)
            else:
                # done loading, now update
                # update others
                loss, acc, _ = run_iter(model, optimizer, params, batch,
                                        is_training=True, train_parser=False)
                loss_comp.append(loss.item())
                # update policy
                assert len(mem.mem) == args.K
                for batch_replay in mem.mem:
                    loss, acc, entropy = run_iter(model, optimizer, params, batch_replay,
                                                  is_training=True, train_parser=True,
                                                  entropy_coef=args.entropy_coef)
                    loss_parser.append(loss.item())
                    entropy_lst.append(entropy.item())
                mem.clear()
                train_iter_tqdm.set_description(f'Ploss {np.mean(loss_parser[-args.K:]):.4f} '
                                                f'Closs {loss_comp[-1]:.4f} '
                                                f'Ent {np.mean(entropy_lst[-args.K:]):.4f}')
                train_iter_tqdm.refresh()

        print(f'Epoch={epoch} '
              f'Loss(parser)={np.mean(loss_parser):.4f} '
              f'Loss(comp)={np.mean(loss_comp):.4f} '
              f'Entropy={np.mean(entropy_lst):.4f}')

        # valid
        with torch.no_grad():
            model.eval()
            valid_losses = []
            valid_accs = []
            for batch_valid in tqdm.tqdm(valid_iter):
                valid_loss, valid_acc, _ = \
                    run_iter(model,
                             optimizer,
                             params,
                             batch_valid,
                             is_training=False,
                             train_parser=False)
                valid_losses.append(valid_loss.item())
                valid_accs.append(valid_acc.item())
            valid_acc = np.mean(valid_accs)
            valid_loss = np.mean(valid_losses)
            # scheduler.step(valid_acc)
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
    parser.add_argument('-lr', type=float, default=1)
    parser.add_argument('-nepoches', type=int, default=1000)
    # parser.add_argument('-parser_batch', type=int, default=50)
    parser.add_argument('-K', type=int, default=15)
    parser.add_argument('--halve-lr-every', type=int, default=2)
    parser.add_argument('-seed', type=int, default=1000)
    parser.add_argument('-entropy_coef', type=float, default=1e-2)
    parser.add_argument('-max_length', type=float, default=100)
    # parser.add_argument('-load', type=str, default='epoch-734loss-1.2901acc-0.566348.pkl')
    parser.add_argument('-load', type=str, default=None)

    args = parser.parse_args()
    basic.init_seed(args.seed)
    try:
        logging.info(f'params:{args}')
        train(args)
    except:
        logging.info(f'params:{args}')

if __name__ == '__main__':
    main()
