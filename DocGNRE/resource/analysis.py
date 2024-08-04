import json

truth = json.load(open('Re-DocRED/dev_revised.json'))
std = {}
tot_evidences = 0
titleset = set([])

title2vectexSet = {}
tmp = []
count = 0
for x in truth:
    entities = set()
    title = x['title']
    titleset.add(title)

    vertexSet = x['vertexSet']
    title2vectexSet[title] = vertexSet

    # if 'labels' not in x:  # official test set from DocRED
    #     continue
    for label in x['labels']:
        # print(label)
        r = label['r']
        h_idx = label['h']
        t_idx = label['t']
        entities.add(label['h'])
        entities.add(label['t'])
    count += len(x['labels'])

    # for label in x['gpt_labels']:
    #     if label['score'] > 0.6: print(label)
    #     r = label['r']
    #     h_idx = label['h']
    #     t_idx = label['t']
    #     std[(title, r, h_idx, t_idx)] = set([])
    #     tmp.append((title, r, h_idx, t_idx))
print(count)