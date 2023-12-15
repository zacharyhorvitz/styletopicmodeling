import pyro
import torch
import json
import pickle
import os 
from prodslda_cls import ProdSLDA

from sklearn.feature_extraction.text import CountVectorizer

MODEL_PATH = '/burg/nlp/users/zfh2000/style_results/pos_bigrams/2023-12-14_17_54_45/model_epoch5_20914.218841552734.pt'
DATA_DIR_PATH = '/burg/nlp/users/zfh2000/style_results/pos_bigrams/maxdf0.5_mindf5_DATA'

with open(os.path.join(DATA_DIR_PATH, 'bows.pickle'), 'rb') as in_file:
    bows = pickle.load(in_file)
        
with open(os.path.join(DATA_DIR_PATH, 'meta_vectorized.pickle'), 'rb') as in_file:
    meta_vectorized = pickle.load(in_file)    

with open(os.path.join(DATA_DIR_PATH, "raw_text.json"), 'r') as in_file:
    raw_text = json.load(in_file)    

with open(os.path.join(DATA_DIR_PATH, "authors_json.json"), 'r') as in_file:
    authors_json = json.load(in_file)    

with open(os.path.join(DATA_DIR_PATH, "meta_feature_to_names.json"), 'r') as in_file:
    meta_feature_to_names = json.load(in_file)

with open(os.path.join(DATA_DIR_PATH, "vectorizer.pickle"), 'rb') as in_file:
    vectorizer = pickle.load(in_file)


import pdb; pdb.set_trace()

pyro.clear_param_store()

prodsdla = torch.load(MODEL_PATH)
prodsdla.eval()


with torch.no_grad():

    for text, author, bow, meta in zip(raw_text['training'], authors_json['training'], bows['training'], meta_vectorized['training']):
        print(author, text)
        # print(text)
        # print(author)
        # print(bow)
        # print(meta)
        # print('------------------')
        # result =  F.softmax(prodsdla.guide(bow.unsqueeze(0), meta.unsqueeze(0))[1])
        # print(result)
        # print('------------------')
        # print('------------------'


# label_to_topic = {}
# label_to_max = {}

# for d, text, encoded_label, label in zip(docs, data['data'], labels, data['labels']):
#     if label not in label_to_topic:
#         label_to_topic[label] = []
#         label_to_max[label] = []
#     # print(d)
#     # print(label)
#     # print('------------------')
#     result =  F.softmax(prodLDA.guide(d.unsqueeze(0), encoded_label.unsqueeze(0))[1])

#     # argmax = torch.argmax(result)
#     # print(argmax)
#     label_to_max[label].append(result[0].detach().cpu().numpy())
#     print(label, result)
#     label_to_topic[label].append((text,result))


# for label in label_to_topic:
#     print(label, np.mean(label_to_max[label]))
