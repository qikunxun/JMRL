import json

with open('dataset/docred/train_distant.json', mode='r') as fd:
    data = json.load(fd)

with open('dataset/docred/train_distant.part3.json', mode='w') as fd:
    json.dump(data[:int(len(data) * 0.9)], fd)

# with open('dataset/docred/train_distant.part2.json', mode='w') as fd:
#     json.dump(data[len(data) // 2:], fd)