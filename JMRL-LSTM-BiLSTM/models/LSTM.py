import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np
from torch.nn.utils import rnn


class LSTM(nn.Module):
	def __init__(self, config, T=2, L=20):
		super(LSTM, self).__init__()
		self.config = config
		self.T = T
		self.L = L
		self.n = config.relation_num

		word_vec_size = config.data_word_vec.shape[0]
		self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])
		self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
		self.word_emb.weight.requires_grad = False

		self.char_emb = nn.Embedding(config.data_char_vec.shape[0], config.data_char_vec.shape[1])
		self.char_emb.weight.data.copy_(torch.from_numpy(config.data_char_vec))
		char_dim = config.data_char_vec.shape[1]
		char_hidden = 100
		self.char_cnn = nn.Conv1d(char_dim,  char_hidden, 5)
		self.coref_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)
		self.ner_emb = nn.Embedding(20, config.entity_type_size, padding_idx=0)

		input_size = config.data_word_vec.shape[1] + config.coref_size + config.entity_type_size #+ char_hidden
		hidden_size = 128

		# self.rnn = EncoderLSTM(input_size, hidden_size, 1, True, True, 1 - config.keep_prob, False)
		# self.linear_re = nn.Linear(hidden_size*2, hidden_size)  # *4 for 2layer
		input_size += char_hidden

		self.diff_w = nn.Parameter(torch.Tensor(self.n, self.T, self.L, 2 * self.n + 1))
		nn.init.kaiming_uniform_(self.diff_w.view(self.n, -1), a=np.sqrt(5))
		self.diff_weights = nn.Parameter(torch.Tensor(self.n, self.L, 1))
		nn.init.kaiming_uniform_(self.diff_weights.view(self.n, -1), a=np.sqrt(5))

		self.rnn = EncoderLSTM(input_size, hidden_size, 1, True, False, 1 - config.keep_prob, False)
		self.linear_re = nn.Linear(hidden_size, hidden_size)  # *4 for 2layer

		self.bili = torch.nn.Bilinear(hidden_size+config.dis_size, hidden_size+config.dis_size, config.relation_num)



		self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)

	def reasoning_by_soft_rules(self, logits):
		n_e = logits.shape[0]
		eye = torch.eye(n_e).to(logits.device)
		input = logits[:, :, :]
		input = torch.cat([input, torch.permute(input, (1, 0, 2)), torch.unsqueeze(eye, dim=-1)], dim=-1)
		all_states = []
		for r in range(self.n):
			cur_states = []
			for t in range(self.T + 1):
				if t == 0:
					w = self.diff_w[r][t]
					one_hot = torch.zeros_like(w.detach())
					if r != 0:
						one_hot[:, 0] = -1e30
						one_hot[:, self.n] = -1e30
					w = torch.softmax(w + one_hot, dim=-1)
					input_cur = input.view(-1, 2 * self.n + 1)
					s_tmp = torch.mm(input_cur, torch.permute(w, (1, 0))).view(n_e, -1, self.L)
					s = s_tmp
					cur_states.append(s)
				if t >= 1 and t < self.T:
					w = self.diff_w[r][t]
					one_hot = torch.zeros_like(w.detach())
					if r != 0:
						one_hot[:, 0] = -1e30
						one_hot[:, self.n] = -1e30
					w = torch.softmax(w + one_hot, dim=-1)
					input_cur = torch.permute(input, (0, 2, 1)).reshape(-1, n_e)
					s_tmp = torch.mm(input_cur, cur_states[t - 1].reshape(n_e, -1))
					s_tmp = s_tmp.view(n_e, 2 * self.n + 1, -1, self.L)
					s_tmp = torch.einsum('mrnl,lr->mnl', s_tmp, w)
					s = s_tmp
					cur_states.append(s)
				if t == self.T:
					weight = torch.tanh(self.diff_weights[r])
					final_state = torch.einsum('mnl,lk->mnk', cur_states[-1], weight).squeeze(dim=-1)
					all_states.append(final_state)
		output = torch.stack(all_states, dim=-1)
		return output

	def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_t_pairs, h_mapping, t_mapping,
				relation_mask, dis_h_2_t, dis_t_2_h):
		para_size, char_size, bsz = context_idxs.size(1), context_char_idxs.size(2), context_idxs.size(0)
		context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size, -1)
		context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)
		h_t_pairs = h_t_pairs + (h_t_pairs == 0).long() - 1  # [batch_size, h_t_limit, 2]

		sent = torch.cat([self.word_emb(context_idxs) , self.coref_embed(pos), self.ner_emb(context_ner)], dim=-1)

		sent = torch.cat([sent, context_ch], dim=-1)


		context_output = self.rnn(sent, context_lens)

		context_output = torch.relu(self.linear_re(context_output))


		start_re_output = torch.matmul(h_mapping, context_output)
		end_re_output = torch.matmul(t_mapping, context_output)

		s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
		t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)
		predict_re = self.bili(s_rep, t_rep)

		n_e = int(np.sqrt(h_t_pairs.shape[1])) + 1
		predictions_new = torch.zeros_like(predict_re)
		for b in range(h_mapping.shape[0]):
			if relation_mask is not None: mask = relation_mask[b]
			indices = h_t_pairs[b].transpose(1, 0).to(predict_re.device)
			input = torch.sigmoid(predict_re[b])
			if relation_mask is not None: input = input * mask.unsqueeze(dim=-1)
			matrix = torch.sparse.FloatTensor(indices.long(), input,
											  torch.Size([n_e, n_e, self.config.relation_num])).to(predict_re.device)
			logits_rule = self.reasoning_by_soft_rules(matrix.to_dense())
			logits_rule = logits_rule.view(-1, self.config.relation_num)
			indices = indices[0] * n_e + indices[1]
			logits_rule = logits_rule[indices.long()]
			predictions_new[b] = logits_rule
		logits_rule_soft = predictions_new + predict_re

		return predict_re, logits_rule_soft


class LockedDropout(nn.Module):
	def __init__(self, dropout):
		super().__init__()
		self.dropout = dropout

	def forward(self, x):
		dropout = self.dropout
		if not self.training:
			return x
		m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
		mask = Variable(m.div_(1 - dropout), requires_grad=False)
		mask = mask.expand_as(x)
		return mask * x

class EncoderRNN(nn.Module):
	def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
		super().__init__()
		self.rnns = []
		for i in range(nlayers):
			if i == 0:
				input_size_ = input_size
				output_size_ = num_units
			else:
				input_size_ = num_units if not bidir else num_units * 2
				output_size_ = num_units
			self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
		self.rnns = nn.ModuleList(self.rnns)
		self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
		self.dropout = LockedDropout(dropout)
		self.concat = concat
		self.nlayers = nlayers
		self.return_last = return_last

		# self.reset_parameters()

	def reset_parameters(self):
		for rnn in self.rnns:
			for name, p in rnn.named_parameters():
				if 'weight' in name:
					p.data.normal_(std=0.1)
				else:
					p.data.zero_()

	def get_init(self, bsz, i):
		return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

	def forward(self, input, input_lengths=None):
		bsz, slen = input.size(0), input.size(1)
		output = input
		outputs = []
		if input_lengths is not None:
			lens = input_lengths.data.cpu().numpy()
		for i in range(self.nlayers):
			hidden = self.get_init(bsz, i)
			output = self.dropout(output)
			if input_lengths is not None:
				output = rnn.pack_padded_sequence(output, lens, batch_first=True)

			output, hidden = self.rnns[i](output, hidden)


			if input_lengths is not None:
				output, _ = rnn.pad_packed_sequence(output, batch_first=True)
				if output.size(1) < slen: # used for parallel
					padding = Variable(output.data.new(1, 1, 1).zero_())
					output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
			if self.return_last:
				outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
			else:
				outputs.append(output)
		if self.concat:
			return torch.cat(outputs, dim=2)
		return outputs[-1]




class EncoderLSTM(nn.Module):
	def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
		super().__init__()
		self.rnns = []
		for i in range(nlayers):
			if i == 0:
				input_size_ = input_size
				output_size_ = num_units
			else:
				input_size_ = num_units if not bidir else num_units * 2
				output_size_ = num_units
			self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
		self.rnns = nn.ModuleList(self.rnns)

		self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
		self.init_c = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])

		self.dropout = LockedDropout(dropout)
		self.concat = concat
		self.nlayers = nlayers
		self.return_last = return_last

		# self.reset_parameters()

	def reset_parameters(self):
		for rnn in self.rnns:
			for name, p in rnn.named_parameters():
				if 'weight' in name:
					p.data.normal_(std=0.1)
				else:
					p.data.zero_()

	def get_init(self, bsz, i):
		return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

	def forward(self, input, input_lengths=None):
		bsz, slen = input.size(0), input.size(1)
		output = input
		outputs = []
		if input_lengths is not None:
			lens = input_lengths.data.cpu().numpy()

		for i in range(self.nlayers):
			hidden, c = self.get_init(bsz, i)

			output = self.dropout(output)
			if input_lengths is not None:
				output = rnn.pack_padded_sequence(output, lens, batch_first=True)

			output, hidden = self.rnns[i](output, (hidden, c))


			if input_lengths is not None:
				output, _ = rnn.pad_packed_sequence(output, batch_first=True)
				if output.size(1) < slen: # used for parallel
					padding = Variable(output.data.new(1, 1, 1).zero_())
					output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
			if self.return_last:
				outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
			else:
				outputs.append(output)
		if self.concat:
			return torch.cat(outputs, dim=2)
		return outputs[-1]

class BiAttention(nn.Module):
	def __init__(self, input_size, dropout):
		super().__init__()
		self.dropout = LockedDropout(dropout)
		self.input_linear = nn.Linear(input_size, 1, bias=False)
		self.memory_linear = nn.Linear(input_size, 1, bias=False)

		self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

	def forward(self, input, memory, mask):
		bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

		input = self.dropout(input)
		memory = self.dropout(memory)

		input_dot = self.input_linear(input)
		memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
		cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
		att = input_dot + memory_dot + cross_dot
		att = att - 1e30 * (1 - mask[:,None])

		weight_one = F.softmax(att, dim=-1)
		output_one = torch.bmm(weight_one, memory)
		weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
		output_two = torch.bmm(weight_two, input)

		return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)
