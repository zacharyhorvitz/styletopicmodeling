from tqdm import tqdm

import torch


# ----- Posterior Latents -----

def top_beta_document(model, vectorizer, top_k=20):
    betas_document = model.beta_document()
    features_to_betas = {}
    idx_to_name = {v:k for k,v in vectorizer.vocabulary_.items()}
    for feature, logits in betas_document.items():
        features_to_betas[feature] = []
        num_features = logits.shape[0]
        top_results = torch.topk(logits, top_k, dim=-1)
        
        ids = top_results.indices.cpu().numpy()
        values = top_results.values.cpu().numpy()
        
        for i in tqdm(range(num_features)):
            features_to_betas[feature].append({'values':values[i], 'top':[idx_to_name[idx] for idx in ids[i]]})
                
    return features_to_betas

def top_beta_meta(model, meta_feature_to_names, top_k=20):
    betas_metas = model.beta_meta()
    features_to_betas = {}
    for feature, logits in betas_metas.items():
        idx_to_name = {i:k for i,k in enumerate(meta_feature_to_names[feature])}
        features_to_betas[feature] = []
        num_features = logits.shape[0]
        top_results = torch.topk(logits, min(top_k, logits.shape[1]), dim=-1)
        ids = top_results.indices.cpu().numpy()
        values = top_results.values.cpu().numpy()
        for i in tqdm(range(num_features)):
            features_to_betas[feature].append({'values':values[i], 'top':[idx_to_name[idx] for idx in ids[i]]})
        
    return features_to_betas 

# ----- Reconstruction Perplexity -----

def calc_doc_perp(prod_slda, eval_dl, device, use_style = True):
    total_ce, total_count = 0., 0.
    for batch in tqdm(eval_dl):
        for key in batch.keys():
            if isinstance(batch[key], dict):
                for key2 in batch[key].keys():
                    batch[key][key2] = batch[key][key2].to(device)
            else:
                batch[key] = batch[key].to(device)

        ce, count = prod_slda.doc_reconstruct_ce(batch, use_style = use_style)
        total_ce += ce
        total_count += count
    return torch.exp(total_ce/total_count).item()

def calc_style_perp(prod_slda, eval_dl, device):
    total_ce, total_count = 0., 0.
    for batch in tqdm(eval_dl):
        for key in batch.keys():
            if isinstance(batch[key], dict):
                for key2 in batch[key].keys():
                    batch[key][key2] = batch[key][key2].to(device)
            else:
                batch[key] = batch[key].to(device)

        ce, count = prod_slda.style_reconstruct_ce(batch)
        total_ce += ce
        total_count += count
    return torch.exp(total_ce/total_count).item()

def full_perp_eval(prod_slda, eval_dl, device):
    doc_perp      = calc_doc_perp(prod_slda, eval_dl, device)
    doc_only_perp = calc_doc_perp(prod_slda, eval_dl, device, use_style = False)
    style_perp    = calc_style_perp(prod_slda, eval_dl, device)
    
    return doc_perp, doc_only_perp, style_perp


# ----- Topic Metrics -----
def single_topic_coherence(topic, cooc_mat, sing_freqs):
    eps = 1e-9
    
    # Get idxs of all term combinations within topic as flat lists
    idx_0 = topic.repeat(topic.shape[0] - 1)
    idx_1 = circulant(topic[::-1])[::-1][:, 1:].flatten()
    term_cooc = np.array(cooc_mat[idx_0, idx_1])
    # print(term_cooc)
    
    # Calc coherence
    numer = np.log(term_cooc + eps) - (np.log(sing_freqs[idx_0]) + np.log(sing_freqs[idx_1]))
    denom = -np.log(term_cooc + eps)
    coh = numer/denom
    
    # Safely replace non coocurring terms with -1's
    coh[cooc_mat[idx_0, idx_1] == 0] = 0
    
    return np.mean(coh)
    
def topic_coherence(topics, bows):
    pres_bows = (bows > 0).astype(int)
    cooc_mat = (pres_bows.T @ pres_bows)/pres_bows.shape[0]
    sing_freqs = cooc_mat.diagonal()
    
    cohs = []
    for topic in topics:
        cohs.append(single_topic_coherence(topic, cooc_mat, sing_freqs))
        
    return np.mean(cohs), cohs

def topic_diversity(topics):
    uniq_cnt = np.unique(topics).shape[0]
    total_cnt = np.prod(topics.shape)
    return uniq_cnt/total_cnt
    
def between_topic_diversity(topics1, topics2):
    uniq_1 = np.unique(topics1)
    uniq_2 = np.unique(topics2)
    inter_uniq = np.unique(np.array([topics1, topics2]))
    
    return inter_uniq.shape[0]/(uniq_1.shape[0] + uniq_2.shape[0])

