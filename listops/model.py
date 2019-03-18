from torch import nn
from torch.nn import init
from nets.colt import CoopLatentTree

class ListopsModel(nn.Module):

    def __init__(self, nclasses, nwords, edim, hdim):
        super(ListopsModel, self).__init__()

        self.nclasses = nclasses
        self.nwords = nwords
        self.edim = edim
        self.hdim = hdim

        self.embedding = nn.Embedding(num_embeddings=nwords,
                                      embedding_dim=edim)

        self.encoder = CoopLatentTree(word_dim=edim,
                                      hidden_dim=hdim,
                                      gumbel_temperature=1)
        self.clf = nn.Linear(hdim, nclasses)

    def reset_parameters(self):
        init.normal_(self.embedding.weight.data, mean=0, std=0.01)
        self.encoder.reset_parameters()
        self.clf.reset_parameters()

    def train_parser(self):
        self.embedding.weight.requires_grad = False
        self.clf.weight.requires_grad = False
        self.clf.bias.requires_grad = False
        self.encoder.train_parser()

    def train_composition(self):
        self.embedding.weight.requires_grad = True
        self.clf.weight.requires_grad = True
        self.clf.bias.requires_grad = True
        self.encoder.train_composition()

    def forward(self, inp, length):
        embeds = self.embedding(inp)
        rep, _ = self.encoder(input=embeds, length=length)
        logits = self.clf(rep)
        return logits
