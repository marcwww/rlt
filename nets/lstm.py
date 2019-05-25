import math

import torch
from torch import nn
from torch.nn import init

from . import basic


class LSTM(nn.Module):

    def __init__(self,
                 word_dim,
                 hidden_dim,
                 use_leaf_rnn,
                 intra_attention,
                 bidirectional,
                 pooling='final'):
        super(LSTM, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.use_leaf_rnn = use_leaf_rnn
        self.intra_attention = intra_attention
        self.bidirectional = bidirectional
        self.pooling = pooling

        assert not (self.bidirectional and not self.use_leaf_rnn)
        # assert not (self.pooling is not None and self.intra_attention)

        if use_leaf_rnn:
            self.leaf_rnn_cell = nn.LSTMCell(
                input_size=word_dim, hidden_size=hidden_dim)
            if bidirectional:
                self.leaf_rnn_cell_bw = nn.LSTMCell(
                    input_size=word_dim, hidden_size=hidden_dim)
        else:
            self.word_linear = nn.Linear(in_features=word_dim,
                                         out_features=2 * hidden_dim)

        if pooling == 'lstm':
            self.pooling_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_leaf_rnn:
            init.kaiming_normal_(self.leaf_rnn_cell.weight_ih.data)
            init.orthogonal_(self.leaf_rnn_cell.weight_hh.data)
            init.constant_(self.leaf_rnn_cell.bias_ih.data, val=0)
            init.constant_(self.leaf_rnn_cell.bias_hh.data, val=0)
            # Set forget bias to 1
            self.leaf_rnn_cell.bias_ih.data.chunk(4)[1].fill_(1)
            if self.bidirectional:
                init.kaiming_normal_(self.leaf_rnn_cell_bw.weight_ih.data)
                init.orthogonal_(self.leaf_rnn_cell_bw.weight_hh.data)
                init.constant_(self.leaf_rnn_cell_bw.bias_ih.data, val=0)
                init.constant_(self.leaf_rnn_cell_bw.bias_hh.data, val=0)
                # Set forget bias to 1
                self.leaf_rnn_cell_bw.bias_ih.data.chunk(4)[1].fill_(1)
        else:
            init.kaiming_normal_(self.word_linear.weight.data)
            init.constant_(self.word_linear.bias.data, val=0)

        if self.pooling == 'lstm':
            init.kaiming_normal_(self.pooling_lstm.weight_ih_l0.data)
            init.orthogonal_(self.pooling_lstm.weight_hh_l0.data)
            init.constant_(self.pooling_lstm.bias_ih_l0.data, val=0)
            init.constant_(self.pooling_lstm.bias_hh_l0.data, val=0)
            # Set forget bias to 1
            self.pooling_lstm.bias_ih_l0.data.chunk(4)[1].fill_(1)

    @staticmethod
    def update_state(old_state, new_state, done_mask):
        old_h, old_c = old_state
        new_h, new_c = new_state
        done_mask = done_mask.float().unsqueeze(1).unsqueeze(2)
        h = done_mask * new_h + (1 - done_mask) * old_h[:, :-1, :]
        c = done_mask * new_c + (1 - done_mask) * old_c[:, :-1, :]
        return h, c

    def select_composition(self, old_state, new_state, mask, indices_given):
        new_h, new_c = new_state  # new_x: (bsz, cur_len, hdim)
        old_h, old_c = old_state  # old_x: (bsz, cur_len + 1, hdim)
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        old_c_left, old_c_right = old_c[:, :-1, :], old_c[:, 1:, :]
        comp_weights = (self.comp_query * new_h).sum(-1)  # self.comp_query: (hdim,)
        comp_weights = comp_weights / math.sqrt(self.hidden_dim)  # comp_weights: (bsz, cur_len)
        if indices_given is not None:
            select_mask = basic. \
                convert_to_one_hot(indices_given, comp_weights.shape[1]).float()
        else:
            if self.training:
                select_mask = basic.st_gumbel_softmax(
                    logits=comp_weights, temperature=self.gumbel_temperature,
                    mask=mask)  # mask: (bsz, cur_len), select_mask: (bsz, cur_len)
            else:
                select_mask = basic.greedy_select(logits=comp_weights, mask=mask)
                select_mask = select_mask.float()
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
        selected_h = (select_mask_expand * new_h).sum(1)
        return new_h, new_c, select_mask, selected_h

    def forward(self, input, length, tree=None):
        max_depth = input.size(1)
        length_mask = basic.sequence_mask(sequence_length=length,
                                          max_length=max_depth)  # length_mask: (bsz, seq_len)

        if self.use_leaf_rnn:
            hs = []
            cs = []
            batch_size, max_length, _ = input.size()
            zero_state = input.data.new_zeros(batch_size, self.hidden_dim)
            h_prev = c_prev = zero_state
            for i in range(max_length):
                h, c = self.leaf_rnn_cell(
                    input=input[:, i, :], hx=(h_prev, c_prev))
                hs.append(h)
                cs.append(c)
                h_prev = h
                c_prev = c
            hs = torch.stack(hs, dim=1)
            cs = torch.stack(cs, dim=1)

            if self.bidirectional:
                hs_bw = []
                cs_bw = []
                h_bw_prev = c_bw_prev = zero_state
                lengths_list = list(length.data)
                input_bw = basic.reverse_padded_sequence(
                    inputs=input, lengths=lengths_list, batch_first=True)
                for i in range(max_length):
                    h_bw, c_bw = self.leaf_rnn_cell_bw(
                        input=input_bw[:, i, :], hx=(h_bw_prev, c_bw_prev))
                    hs_bw.append(h_bw)
                    cs_bw.append(c_bw)
                    h_bw_prev = h_bw
                    c_bw_prev = c_bw
                hs_bw = torch.stack(hs_bw, dim=1)
                cs_bw = torch.stack(cs_bw, dim=1)
                hs_bw = basic.reverse_padded_sequence(
                    inputs=hs_bw, lengths=lengths_list, batch_first=True)
                cs_bw = basic.reverse_padded_sequence(
                    inputs=cs_bw, lengths=lengths_list, batch_first=True)
                hs = torch.cat([hs, hs_bw], dim=2)
                cs = torch.cat([cs, cs_bw], dim=2)
            state = (hs, cs)
        else:
            state = self.word_linear(input)
            state = state.chunk(chunks=2, dim=2)
        # state: (bsz, seq_len, hdim or hdim * 2) * 2

        nodes = state[0] * length_mask.unsqueeze(-1).float()  # (bsz, seq_len, hdim or hdim * 2)
        h, c = state  # h/c: (bsz, 1, hdim)
        if self.intra_attention:
            att_mask = length_mask.float()
            # nodes: (batch_size, num_tree_nodes, hidden_dim)
            # nodes_mean: (batch_size, hidden_dim, 1)
            nodes_mean = nodes.mean(1).squeeze(1).unsqueeze(2)
            # att_weights: (batch_size, num_tree_nodes)
            att_weights = torch.bmm(nodes, nodes_mean).squeeze(2)
            att_weights = basic.masked_softmax(
                logits=att_weights, mask=att_mask)
            # att_weights_expand: (batch_size, num_tree_nodes, hidden_dim)
            att_weights_expand = att_weights.unsqueeze(2).expand_as(nodes)
            # h: (batch_size, 1, 2 * hidden_dim)
            h = (att_weights_expand * nodes).sum(1)
        else:
            if self.pooling == 'lstm':
                os, _ = self.pooling_lstm(nodes)
                h = os[range(os.shape[0]), length - 1]

            elif self.pooling == 'mean':
                h = nodes.sum(dim=1).div(length.unsqueeze(-1).float())

            elif self.pooling == 'final':
                h = nodes[range(nodes.shape[0]), length - 1]

            else:
                raise NotImplementedError

        return h.squeeze(1)
