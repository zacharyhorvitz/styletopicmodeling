import json
import pickle
import os

import numpy as np
from scipy import sparse

from tqdm import tqdm

from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer

from .preprocess import *


def load_data(save_dir):
    '''
        Load a pre-processed dataset from the given directory
    '''
    
    with open(os.path.join(save_dir, 'bows.pickle'), 'rb') as in_file:
        bows = pickle.load(in_file)
        
    with open(os.path.join(save_dir, 'meta_vectorized.pickle'), 'rb') as in_file:
        meta_vectorized = pickle.load(in_file)    
    
    with open(os.path.join(save_dir,"vectorizer.pickle"), 'rb') as in_file:
        vectorizer = pickle.load(in_file)   

    with open(os.path.join(save_dir, "raw_text.json"), 'r') as in_file:
        raw_text = json.load(in_file)    

    with open(os.path.join(save_dir, "authors_json.json"), 'r') as in_file:
        authors_json = json.load(in_file)    

    with open(os.path.join(save_dir, "meta_feature_to_names.json"), 'r') as in_file:
        meta_feature_to_names = json.load(in_file)
        
    return bows, meta_vectorized, vectorizer, raw_text, authors_json, meta_feature_to_names

def prepare_data(data_src, synthetic_src, splits_path, meta_features, save_dir, 
                 max_df = 0.7, min_df = 20):
    '''
        Preprocess dataset and store in given directory
    '''
    
    with open(data_src, 'r') as in_file:
        data = json.load(in_file)
    
    with open(synthetic_src, 'r') as in_file:
        synthetic_data = json.load(in_file)
    
    data = data + synthetic_data
    
    with open(splits_path, 'r') as in_file:
        splits = json.load(in_file)
    
    for x in data:
        clean_up_features(x)
            
    meta_feature_to_names = {}
    
    for key in meta_features:
        meta_feature_to_names[key] = get_possible_values(data, key)
    
    extracted_features = [extract_features(d, meta_feature_to_names) for d in data]
        
    training = []
    holdout = []

    author_labels = {'training':[], 'holdout':[]}

    raw_text = {'training':[], 'holdout':[]}
    
    authors = [d['info']['from'] for d in data]
    texts = [d['text'] for d in data]
    for author, t, d  in zip(authors, texts, extracted_features):
        if author.startswith('gpt') or author in splits['train']:
            training.append(d)
            author_labels['training'].append(author)
            raw_text['training'].append(t)
            
        else:
            holdout.append(d)
            author_labels['holdout'].append(author)
            assert author in splits['test']
            raw_text['holdout'].append(t)
    
    
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english')
    vectorizer.fit([d['text'] for d in training])
    
    bows = {}

    meta_vectorized = {}
    
    for split_name, data_split in zip(['training','holdout'], [training, holdout]):
        bows[split_name] = vectorizer.transform([d['text'] for d in data_split])
        meta_vectorized[split_name] = {}
        for key in tqdm(sorted(data_split[0].keys())):
            if key in ['text']: continue
            meta_vectorized[split_name][key] = sparse.csr_matrix(np.stack([d[key] for d in data_split]))
            assert(bows[split_name].shape[0] == meta_vectorized[split_name][key].shape[0])

    os.makedirs(save_dir, exist_ok=False)
    pickle.dump(vectorizer, open(os.path.join(save_dir,"vectorizer.pickle"), "wb"))
    pickle.dump(bows, open(os.path.join(save_dir,"bows.pickle"), "wb"))
    pickle.dump(meta_vectorized, open(os.path.join(save_dir,"meta_vectorized.pickle"), "wb"))
    
    json.dump(raw_text, open(os.path.join(save_dir,"raw_text.json"), "w"))
    json.dump(author_labels, open(os.path.join(save_dir,"authors_json.json"), "w"))
    json.dump(meta_feature_to_names, open(os.path.join(save_dir,"meta_feature_to_names.json"), "w"))
    
    print(f'Saved preprocessed data to {save_dir}')
        
    return bows, meta_vectorized, vectorizer, raw_text, authors_json, meta_feature_to_names

def batch_to_device(batch, device):
    '''
        Move batch of DocMetaData to given device
    '''
    
    bow = batch['bow'].to(device)
    meta = {k:v.to(device) for k,v in batch['meta'].items()}
    
    new_batch = {'bow': bow, 'meta': meta}
    
    if 'bow_h1' in batch.keys() and 'bow_h2' in batch.keys():
        bow_h1 = batch['bow_h1'].to(device)
        bow_h2 = batch['bow_h2'].to(device)
        
        new_batch['bow_h1'] = bow_h1
        new_batch['bow_h2'] = bow_h2
    
    return new_batch

class DocMetaData(Dataset):
    
    def __init__(self, bows, metas, 
                 split_halves = False, perc_obs = 0.5,
                 dtype = np.float32):
        '''
            Dataset containing BOW representations of documents and associated style metadata
        '''
        
        self.bows = bows
        self.metas = metas
        
        self.split_halves = split_halves
        self.perc_obs = perc_obs
        
        self.dtype = dtype
        
        if self.split_halves:
            bowh_counts = (bows.toarray().sum(axis= -1) * self.perc_obs).astype(int).tolist()
    
            self.bows_h1, self.bows_h2 = [], []
            for bow, bcount in zip(bows, bowh_counts):
                h1 = bow.toarray().copy().squeeze()
                for i in range(bcount):
                    idx = np.random.choice(h1.squeeze().nonzero()[0])
                    h1[idx] -= 1
                h2 = (bow.toarray() - h1).squeeze()

                self.bows_h1.append(h1)
                self.bows_h2.append(h2)
            
            self.bows_h1 = sparse.csr_matrix(self.bows_h1)
            self.bows_h2 = sparse.csr_matrix(self.bows_h2)
        
    def __len__(self):
        return self.bows.shape[0]
    
    def __getitem__(self, idx):
        
        bow = self.bows[idx].toarray().astype(self.dtype)[0]
        
        meta = {key:self.metas[key][idx].toarray().astype(self.dtype)[0] for key in self.metas}
        
        batch = {
            'bow': bow,
            'meta': meta,
        }
        
        if self.split_halves:
            batch['bow_h1'] = self.bows_h1[idx].toarray().astype(self.dtype)[0]
            batch['bow_h2'] = self.bows_h2[idx].toarray().astype(self.dtype)[0]
        
        return batch