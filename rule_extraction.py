import sys
import json
import torch
beam_size = int(sys.argv[2])

def reverse(x2id):
    id2x = {}
    for x in x2id:
        id2x[x2id[x]] = x
    return id2x

def get_beam(indices, t, beam_size, r, T, all_indices):
    beams = indices[:, :beam_size]
    for l in range(indices.shape[0]):
        index = indices[l]
        j = 0
        for i in range(index.shape[-1]):
            beams[l][j] = index[i]
            j += 1
            if j == beam_size: break

    return beams

def get_states(indices, scores):
    states = torch.zeros(indices.shape)
    for l in range(indices.shape[0]):
        states[l] = torch.index_select(scores[l], -1, indices[l])
    return states

def transform_score(x, T):
    one = torch.autograd.Variable(torch.Tensor([1]))
    zero = torch.autograd.Variable(torch.Tensor([0]).detach())
    return torch.minimum(torch.maximum(x / T, zero), one)

def analysis(model_save_path, id2relation):
    print('=' * 50)
    model = torch.load(model_save_path, map_location=torch.device('cpu'))
    w = model['diff_w']
    print(w.shape)
    weight = model['diff_weights']
    T = 3
    n = 66
    L = 20
    for target in range(w.shape[0]):
        one_hot = torch.zeros_like(w[target][-1])
        if target != 0:
            one_hot[:, 0] = -1e30
            one_hot[:, n] = -1e30
        w_ = torch.log_softmax(w[target][-1] + one_hot, dim=-1)
        scores = w_
        indices_order = torch.argsort(scores, dim=-1, descending=True)
        all_indices = []
        indices = get_beam(indices_order, T - 2, beam_size, 2 * n, T, all_indices)
        states = get_states(indices, scores)
        all_indices.append(indices)
        for t in range(T - 3, -1, -1):
            one_hot = torch.zeros_like(w[target][-1])
            if target != 0:
                one_hot[:, 0] = -1e30
                one_hot[:, n] = -1e30
            w_ = torch.log_softmax(w[target][t] + one_hot, dim=-1)
            scores = states.unsqueeze(dim=-1) + w_.unsqueeze(dim=1)

            scores = scores.view(L, -1)
            indices_order = torch.argsort(scores, dim=-1, descending=True)
            topk_indices = get_beam(indices_order, t, beam_size, 2 * n, T, all_indices)
            states = get_states(topk_indices, scores)
            all_indices.append(topk_indices)
        outputs = torch.zeros(L, T - 1, beam_size).long()
        p = torch.zeros(L, beam_size).long()
        for beam in range(beam_size):
            p[:, beam] = beam
        for t in range(T - 1):
            for l in range(L):
                for beam in range(beam_size):
                    c = int(all_indices[T - t - 2][l][p[l][beam]] % (2 * n + 1))
                    outputs[l][t][beam] = c
                    p_new = int(all_indices[T - t - 2][l][p[l][beam]] / (2 * n + 1))
                    p[l][beam] = p_new
        all_rules = []
        for l in range(L):
            rule = '{}(x, y)<-'.format(id2relation[target])
            rules = [rule] * beam_size
            counts = torch.zeros(L, beam_size)
            for beam in range(beam_size):
                y = ''
                for t in range(T - 2, -1, -1):
                    c = int(outputs[l][t][beam])
                    if c < 2 * n:
                        tmp = id2relation[c]
                        x = 'x'
                        if counts[l][beam] > 0: x = 'z_{}'.format(int(counts[l][beam]) - 1)
                        y = 'z_{}'.format(int(counts[l][beam]))
                        if t == 0 or (t > 0 and outputs[l][t - 1][beam] == 2 * n): y = 'y'
                        flag = tmp + '({}, {})'.format(x, y)
                        counts[l][beam] += 1
                    else:
                        identity = 'x'
                        if t != T - 2: identity = y
                        flag = 'Identity({}, {})'.format(identity, identity)
                    end = ''
                    if t > 0: end = ' ∧ '
                    rules[beam] += flag + end

                if not rules[beam].endswith('y)'):
                    output_tmp = rules[beam].split()
                    rules[beam] = rules[beam].replace(output_tmp[-1][:-1], 'y')
            all_rules.append(rules)

        ids_sort = torch.argsort(weight[target].squeeze(dim=-1), descending=True)
        for i, ids in enumerate(ids_sort):
            print('Rank: {}, id: {}, Rule: {}, Weight: {}'.format((i + 1), ids, all_rules[int(ids)], float(torch.tanh(weight[target][int(ids)]))))



if __name__ == '__main__':
    with open('dataset_dwie/meta/rel2id.json') as fd:
        rel2id = json.load(fd)
    id2relation = {}
    for rel in rel2id:
        id2relation[rel2id[rel]] = rel
    length = len(id2relation)
    for id in id2relation.copy():
        id2relation[id + length] = 'INV' + id2relation[id]
    print(len(id2relation))
    analysis(sys.argv[1], id2relation)