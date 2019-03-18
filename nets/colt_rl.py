import math

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.distributions import Categorical

from . import basic


class BinaryTreeLSTMLayer(nn.Module):

    def __init__(self, hidden_dim):
        super(BinaryTreeLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = nn.Linear(in_features=2 * hidden_dim,
                                     out_features=5 * hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.comp_linear.weight.data)
        init.constant_(self.comp_linear.bias.data, val=0)

    def forward(self, l=None, r=None):
        """
        Args:
            l: A (h_l, c_l) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
            r: A (h_r, c_r) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
        Returns:
            h, c: The hidden and cell state of the composed parent,
                each of which has the size
                (batch_size, max_length - 1, hidden_dim).
        """

        hl, cl = l
        hr, cr = r
        hlr_cat = torch.cat([hl, hr], dim=2)
        treelstm_vector = self.comp_linear(hlr_cat)
        i, fl, fr, u, o = treelstm_vector.chunk(chunks=5, dim=2)
        c = (cl * (fl + 1).sigmoid() + cr * (fr + 1).sigmoid()
             + u.tanh() * i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c


class CoopLatentTreeRL(nn.Module):

    def __init__(self, word_dim, hidden_dim, gumbel_temperature):
        super(CoopLatentTreeRL, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim

        self.gumbel_temperature = gumbel_temperature
        self.word_linear = nn.Linear(in_features=word_dim,
                                     out_features=2 * hidden_dim)

        self.treelstm_layer = BinaryTreeLSTMLayer(hidden_dim)
        self.comp_query = nn.Parameter(torch.FloatTensor(hidden_dim))
        self.reset_parameters()
        self.train_parser_flag = True

    def reset_parameters(self):
        init.kaiming_normal_(self.word_linear.weight.data)
        init.constant_(self.word_linear.bias.data, val=0)
        self.treelstm_layer.reset_parameters()
        init.normal_(self.comp_query.data, mean=0, std=0.01)

    def train_parser(self):
        def fix(module: nn.Module):
            for param in module.parameters():
                param.requires_grad = False

        self.apply(fix)
        self.comp_query.requires_grad = True
        self.train_parser_flag = True

    def train_composition(self):
        def dyn(module: nn.Module):
            for param in module.parameters():
                param.requires_grad = True

        self.apply(dyn)
        self.comp_query.requires_grad = False
        self.train_parser_flag = False

    @staticmethod
    def update_state(old_state, new_state, done_mask):
        old_h, old_c = old_state
        new_h, new_c = new_state
        done_mask = done_mask.float().unsqueeze(1).unsqueeze(2)
        h = done_mask * new_h + (1 - done_mask) * old_h[:, :-1, :]
        c = done_mask * new_c + (1 - done_mask) * old_c[:, :-1, :]
        return h, c

    def select_composition(self, old_state, new_state, mask,
                           self_critic=False,
                           indices_given=None):
        new_h, new_c = new_state  # new_x: (bsz, cur_len, hdim)
        old_h, old_c = old_state  # old_x: (bsz, cur_len + 1, hdim)
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        old_c_left, old_c_right = old_c[:, :-1, :], old_c[:, 1:, :]
        comp_weights = (self.comp_query * new_h).sum(-1)  # self.comp_query: (hdim,)
        comp_weights = comp_weights / math.sqrt(self.hidden_dim)  # comp_weights: (bsz, cur_len)
        comp_probs = basic.masked_softmax(comp_weights, mask)
        comp_dist = Categorical(comp_probs)
        if indices_given is not None:
            select_mask = basic.convert_to_one_hot(indices_given, comp_probs.shape[1]).float()
            log_prob = comp_dist.log_prob(indices_given)
            indices = indices_given
        else:
            if self.training and not self_critic:
                indices_sampled = comp_dist.sample()
                select_mask = basic.convert_to_one_hot(indices_sampled, comp_probs.shape[1]).float()
                log_prob = comp_dist.log_prob(indices_sampled)
                indices = indices_sampled
            else:
                indices_argmax = comp_probs.max(1)[1]
                select_mask = basic.convert_to_one_hot(indices_argmax, comp_probs.shape[1]).float()
                log_prob = comp_dist.log_prob(indices_argmax)
                indices = indices_argmax

        select_mask_expand = select_mask.unsqueeze(2).expand_as(new_h)
        select_mask_cumsum = select_mask.cumsum(1)
        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(2).expand_as(old_h_left)
        right_mask = select_mask_cumsum - select_mask
        right_mask_expand = right_mask.unsqueeze(2).expand_as(
            old_h_right)  # here left/right_mask does not include the selected point
        new_h = (select_mask_expand * new_h
                 + left_mask_expand * old_h_left
                 + right_mask_expand * old_h_right)
        new_c = (select_mask_expand * new_c
                 + left_mask_expand * old_c_left
                 + right_mask_expand * old_c_right)
        # selected_h = (select_mask_expand * new_h).sum(1)
        return new_h, new_c, left_mask_expand, right_mask_expand,\
               indices, log_prob, comp_dist.entropy()

    def forward(self, input, length, self_critic=False, tree=None):
        max_depth = input.size(1)
        length_mask = basic.sequence_mask(sequence_length=length,
                                          max_length=max_depth)  # length_mask: (bsz, seq_len)
        indices_select = []
        state = self.word_linear(input)
        state = state.chunk(chunks=2, dim=2)
        h, c = state
        l = (h[:, :-1, :], c[:, :-1, :])
        r = (h[:, 1:, :], c[:, 1:, :])
        state_comp = self.treelstm_layer(l=l, r=r)

        log_prob_sum = 0
        entropy_sum = 0
        for i in range(max_depth - 1):

            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_h, new_c, left_mask, right_mask, indices, log_prob, entropy = \
                    self.select_composition(
                        old_state=state,
                        new_state=state_comp,
                        mask=length_mask[:, i + 1:],
                        indices_given=tree[i] if tree else None)  # select_mask: (bsz, cur_len), i.e. select_dist over
                #  constituents at each time step
                state_new = (new_h, new_c)
                indices_select.append(indices)
                log_prob_sum += log_prob
                entropy_sum += entropy

                done_mask = length_mask[:, i + 1]
                state = self.update_state(old_state=state, new_state=state_new,
                                          done_mask=done_mask)  # this step is to retain the finised final hiddens
                # TODO: update state_comp with treelstm_layer and state
                h, c = state
                bsz, seq_len, hdim = h.shape
                indices_l = torch.stack([indices - 1, indices], dim=1).clamp_(min=0, max=seq_len-1).\
                    unsqueeze(-1).expand(bsz, 2, hdim)
                indices_r = torch.stack([indices, indices + 1], dim=1).clamp_(min=0, max=seq_len-1).\
                    unsqueeze(-1).expand(bsz, 2, hdim)
                l = (h.gather(dim=1, index=indices_l), c.gather(dim=1, index=indices_l))
                r = (h.gather(dim=1, index=indices_r), c.gather(dim=1, index=indices_r))
                state_comp_singtwo = self.treelstm_layer(l=l, r=r) # state_comp_singtwo: (bsz, 2, hdim)
                h_comp, c_comp = state_comp
                l_comp = (h_comp[:, :-1] * left_mask[:, :-1], c_comp[:, :-1] * left_mask[:, :-1])
                r_comp = (h_comp[:, 1:] * right_mask[:, 1:], c_comp[:, 1:] * right_mask[:, 1:])
                state_comp = (l_comp[0] + r_comp[0], l_comp[1] + r_comp[1])
                mask_le = indices.le(seq_len - 2).unsqueeze(-1)
                mask_ge = (indices - 1).ge(0).unsqueeze(-1)
                state_comp[0][range(bsz), torch.clamp(indices, min=0, max=seq_len-2)] += state_comp_singtwo[0][:, 1].masked_fill_(mask_le, 0)
                state_comp[1][range(bsz), torch.clamp(indices, min=0, max=seq_len-2)] += state_comp_singtwo[1][:, 1].masked_fill_(mask_le, 0)

                state_comp[0][range(bsz), torch.clamp(indices - 1, min=0, max=seq_len-2)] += state_comp_singtwo[0][:, 0].masked_fill_(mask_ge, 0)
                state_comp[1][range(bsz), torch.clamp(indices - 1, min=0, max=seq_len-2)] += state_comp_singtwo[1][:, 0].masked_fill_(mask_ge, 0)

            else:
                done_mask = length_mask[:, i + 1]
                state = self.update_state(old_state=state, new_state=state_comp,
                                          done_mask=done_mask)  # this step is to retain the finised final hiddens
                h, c = state
                l = (h[:, :-1], c[:, :-1])
                r = (h[:, 1:], c[:, 1:])
                state_comp = self.treelstm_layer(l=l, r=r)

        h, c = state  # h/c: (bsz, 1, hdim)

        assert h.size(1) == 1 and c.size(1) == 1

        return h.squeeze(1), c.squeeze(1), indices_select, log_prob_sum, entropy_sum.div(length.float())
