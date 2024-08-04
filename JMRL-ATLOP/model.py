import torch
import torch.nn as nn
from long_seq import process_long_input
from losses import ATLoss
import numpy as np



class DocREModel(nn.Module):
    def __init__(self, config, model, dataset='dwie', emb_size=768, block_size=64, num_labels=-1,
                 temperature=1, lambda_al=1, T=2, L=20, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        self.model = model
        self.hidden_size = config.hidden_size
        self.temperature = temperature
        self.T = T
        self.L = L
        self.n = config.num_labels
        self.loss_fnt = ATLoss()

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.diff_w = nn.Parameter(torch.Tensor(self.n, self.T, self.L, 2 * self.n + 1))
        nn.init.kaiming_uniform_(self.diff_w.view(self.n, -1), a=np.sqrt(5))
        self.diff_weights = nn.Parameter(torch.Tensor(self.n, self.L, 1))
        nn.init.kaiming_uniform_(self.diff_weights.view(self.n, -1), a=np.sqrt(5))

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        self.lambda_al = lambda_al


    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = torch.einsum("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def add_anti_to_labels(self, labels: torch.Tensor, hts: list) -> torch.Tensor:
        anti_labels = torch.zeros(size=(labels.size(0), labels.size(1) - 1)).to(labels)
        past_entity_pairs = 0
        for hts_one_doc in hts:
            for index, [h, t] in enumerate(hts_one_doc):
                if labels[index + past_entity_pairs, 0] == 1:
                    break
                anti_labels[past_entity_pairs + hts_one_doc.index([t, h])] = labels[index + past_entity_pairs, 1:]

            past_entity_pairs += len(hts_one_doc)
        return torch.cat((labels, anti_labels), dim=1)

    def reasoning_by_soft_rules(self, logits):
        n_e = logits.shape[0]
        eye = torch.eye(n_e).to(self.device)
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

    def activation(self, x):
        return torch.minimum(torch.maximum(x, torch.zeros_like(x)), torch.ones_like(x))

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                output_for_LogiRE=False,
                ):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)
        logits_rule_soft = []
        start = 0
        for b in range(len(hts)):
            indices = torch.LongTensor(hts[b]).transpose(1, 0).to(logits)
            n_e = int(np.sqrt(len(hts[b]))) + 1
            end = start + len(hts[b])
            input = torch.softmax(logits[start: end, :], dim=-1)
            matrix = torch.sparse.FloatTensor(indices.long(), input,
                                              torch.Size([n_e, n_e, self.config.num_labels])).to(logits)
            logits_rule = self.reasoning_by_soft_rules(matrix.to_dense())
            logits_rule = logits_rule.view(-1, self.config.num_labels)
            indices = indices[0] * n_e + indices[1]
            logits_rule = logits_rule[indices.long()]
            logits_rule_soft.append(logits_rule)
            start = end
        logits_rule_soft = torch.cat(logits_rule_soft, dim=0) + logits
        if output_for_LogiRE:
            return logits

        output = (self.loss_fnt.get_label(logits_rule_soft, num_labels=self.num_labels),)

        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)

            loss_cls = self.loss_fnt(logits.float(), labels.float())
            loss_rule = self.loss_fnt(logits_rule_soft.float() / 0.2, labels.clone().float())
            loss = self.lambda_al * loss_cls + loss_rule
            loss_dict = {'loss_cls': loss_cls.item(), 'loss_rule': loss_rule.item()}
            print(loss_dict)
            output = (loss.to(sequence_output), loss_dict) + output
        return output
