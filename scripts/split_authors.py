import json
import random
from collections import Counter

DATA_SRC = '/burg/nlp/users/zfh2000/enron_processed.json'

with open(DATA_SRC, 'r') as in_file:
   data = json.load(in_file)

authors = []
for email in data:
    authors.append(email['info']['from'])


authors = sorted(authors)

random.seed(42)

random.shuffle(authors)

train = authors[:int(len(authors)*0.8)]
test = authors[int(len(authors)*0.8):]

print(len(train), len(test))

with open('authors_splits.json', 'w+') as out_file:
    json.dump(dict(
        train=train,
        test=test
    ), out_file)

