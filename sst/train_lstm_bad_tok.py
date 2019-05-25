import argparse
import logging
import os
import tqdm

from tensorboardX import SummaryWriter
from torch.nn import functional as F
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

    def __init__(self, txt, lbl):
        self.txt = txt
        self.lbl = lbl


def load_examples(fname, subtrees=False):
    examples = []
    lens = []
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            txt, _, _, lbl = line.strip().split('\t')
            txt = txt.split()
            examples.append(Example(txt, lbl))
            lens.append(len(txt))

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

    ftrain = 'data/train.txt.tok.dep'
    fvalid = 'data/valid.txt.tok.dep'
    ftest = 'data/test.txt.tok.dep'

    examples_train, len_ave = load_examples(ftrain, subtrees=True)
    examples_valid, _ = load_examples(fvalid, subtrees=False)
    examples_test, _ = load_examples(ftest, subtrees=False)
    train = Dataset(examples_train, fields=[('txt', TXT),
                                            ('lbl', LBL)])
    TXT.build_vocab(train, vectors=args.pretrained)
    LBL.build_vocab(train)
    print(LBL.vocab.itos)
    valid = Dataset(examples_valid, fields=[('txt', TXT),
                                            ('lbl', LBL)])
    test = Dataset(examples_test, fields=[('txt', TXT),
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


def test(model, data_loader, args):
    def run_iter(batch, is_training, latent_tree=True):
        model.train(is_training)
        words, length = batch.txt
        label = batch.lbl
        logits = model(words=words, length=length,
                       tree=None)
        label_pred = logits.max(1)[1]
        accuracy = torch.eq(label, label_pred).float().mean()
        return accuracy

    test_accuracy_sum = 0
    num_test_batches = len(data_loader)
    for valid_batch in data_loader:
        test_accuracy = run_iter(
            batch=valid_batch, is_training=False,
            latent_tree=args.latent_tree)
        test_accuracy_sum += test_accuracy.item()

    test_accuracy = test_accuracy_sum / num_test_batches
    # logging.info(f'test acc {test_accuracy:.4f}')
    return test_accuracy


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
                     dropout_prob=args.dropout,
                     pooling=args.pooling)
    if args.pretrained:
        model.word_embedding.weight.data.set_(TXT.vocab.vectors)
    if args.fix_word_embedding:
        logging.info('Will not update word embeddings')
        model.word_embedding.weight.requires_grad = False
    logging.info(f'Using device {args.gpu}')
    device = torch.device(args.gpu if args.gpu != -1 else 'cpu')
    if args.load is not None:
        model.load_state_dict(torch.load(args.load))
        logging.info(f'Loaded from {args.load}')

    model.to(device)
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

    def run_iter(batch, is_training, latent_tree=True):
        model.train(is_training)
        words, length = batch.txt
        label = batch.lbl
        logits = model(words=words, length=length,
                       tree=None)
        label_pred = logits.max(1)[1]
        accuracy = torch.eq(label, label_pred).float().mean()
        loss = criterion(input=logits, target=label)
        loss_ = F.cross_entropy(input=logits, target=label, reduce=False)
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

    test_acc = test(model, test_loader, args)
    print('Test', test_acc)

    for epoch in range(args.max_epoch):
        loss_train = []
        for batch_iter, train_batch in enumerate(tqdm.tqdm(train_loader)):
            train_loss, train_accuracy = run_iter(
                batch=train_batch, is_training=True,
                latent_tree=args.latent_tree)
            iter_count += 1
            add_scalar_summary(
                summary_writer=train_summary_writer,
                name='loss', value=train_loss, step=iter_count)
            add_scalar_summary(
                summary_writer=train_summary_writer,
                name='accuracy', value=train_accuracy, step=iter_count)
            loss_train.append(train_loss.item())

            if (batch_iter + 1) % validate_every == 0:
                valid_loss_sum = valid_accuracy_sum = 0
                num_valid_batches = len(valid_loader)
                for valid_batch in valid_loader:
                    valid_loss, valid_accuracy = run_iter(
                        batch=valid_batch, is_training=False,
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
                             f'train loss = {np.mean(loss_train):.4f}, '
                             f'valid accuracy = {valid_accuracy:.4f}')
                if valid_accuracy > best_vaild_accuacy:
                    test_acc = test(model, test_loader, args)
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
    parser.add_argument('-load', type=str, default=None)
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
    # parser.add_argument('-pooling', default='final', type=str)
    parser.add_argument('-pooling', default='lstm', type=str)
    args = parser.parse_args()

    basic.init_seed(args.seed)
    logging.info(f'params:{args}')
    # try:
    train(args)
    # except:
    #     logging.info(f'params:{args}')


if __name__ == '__main__':
    main()
