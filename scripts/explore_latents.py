import pyro
import torch
import json
import pickle

from prodslda_cls import ProdSLDA

MODEL_PATH = '/burg/nlp/users/zfh2000/style_results/pos_bigrams/2023-12-14_17_54_45/model_epoch5_20914.218841552734.pt'
DATA_PATG = '/burg/nlp/users/zfh2000/style_results/pos_bigrams/maxdf0.5_mindf5_DATA'
pyro.clear_param_store()

prodsdla = torch.load('prod_slda_saved_model')
prodsdla.eval()
import pdb; pdb.set_trace()


# with torch.no_grad():


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
