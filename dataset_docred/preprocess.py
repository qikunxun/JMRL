import json

with open('rel_info.json') as fd:
    rel_info = json.load(fd)
rel2id = {'NA': 0}
for rel in rel_info:
    rel2id[rel] = len(rel2id)

with open('meta/rel2id.json', mode='w') as fw:
    json.dump(rel2id, fw)

# with open('train_annotated.json', mode='r') as fd:
#     data = json.load(fd)
# all_data = []
# for item in data:
#     for label in item['labels']:
#         r = label['r']
#         hs = item['vertexSet'][label['h']]
#         ts = item['vertexSet'][label['t']]
#         evidences = label['evidence']
#         h_set = set()
#         for h in hs:
#             h_set.add(h['name'])
#         t_set = set()
#         for t in ts:
#             t_set.add(t['name'])
#         for h in h_set:
#             for t in t_set:
#                 print('{}\t{}\t{}'.format(h, rel_info[r].replace(' ', '_'), t))
#                 all_data.append([h, t, r])
#
# with open('ref/train_annotated.fact', mode='w') as fw:
#     json.dump(all_data, fw)