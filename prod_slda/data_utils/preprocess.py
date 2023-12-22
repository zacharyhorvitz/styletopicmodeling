import numpy as np

def get_possible_values(data, key):
    if isinstance(data[0][key], dict):
        return sorted(data[0][key].keys())
    else:
        values = []
        for d in data:
            values.extend(d[key])
        return sorted(set(values))

def clean_up_features(x):
    # emotion:
    if 'unknown' not in x['emotion']:
        if sum(x['emotion'].values()) == 0:
            x['emotion']['unknown'] = 1
        else:
            x['emotion']['unknown'] = 0

    # morph
    x['morph_tags'] = {k:v for k,v in x['gram2vec'].items() if k.startswith('morph')}
    x['punc_tags'] = {k:v for k,v in x['gram2vec'].items() if k.startswith('punctuation')}
    x['sentences'] = {k:v for k,v in x['gram2vec'].items() if k.startswith('sentences')}
    x['pos_bigrams'] = {k:v for k,v in x['gram2vec'].items() if k.startswith('pos_bigrams')}

    if 'none' not in x['punc_tags']:
        if sum(x['punc_tags'].values()) == 0:
            x['punc_tags']['none'] = 1
        else:
            x['punc_tags']['none'] = 0

    if 'none' not in x['sentences']:
        if sum(x['sentences'].values()) == 0:
            x['sentences']['none'] = 1
        else:
            x['sentences']['none'] = 0
            
        
    
def fix_normalization(x):
    # print(x)
    replaced = np.where(x==0, 1, x)
    minimum = np.min(replaced)
    if minimum < 1:
        # print(minimum)
        scale = 1/minimum
        x = (x*scale).astype(int)
    # print(x)
    return x


def extract_features(doc, meta_feature_to_names):
    features = {'text':doc['text']}
    for key, possible in meta_feature_to_names.items():
        if isinstance(doc[key], list):
            features[key] = np.array([doc[key].count(l) for l in possible])
        elif isinstance(doc[key], dict):
            features[key] = np.array([doc[key][l] for l in possible]).astype(float)
        else:
            raise NotImplementedError()

        features[key] = fix_normalization(features[key])
    return features