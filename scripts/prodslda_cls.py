import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam


from torch.utils.data import Dataset


class DocMetaData(Dataset):
    
    def __init__(self, bows, metas, dtype = np.float32):
        self.bows = bows
        self.metas = metas
        
        self.dtype = dtype
        
    def __len__(self):
        return self.bows.shape[0]
    
    def __getitem__(self, idx):
        
        bow = self.bows[idx].toarray().astype(self.dtype)[0]
        
        meta = {key:self.metas[key][idx].toarray().astype(self.dtype)[0] for key in self.metas}
        
        batch = {
            'bow': bow,
            'meta': meta,
        }
        
        return batch

class GeneralEncoder(nn.Module):
    
    def __init__(self, size_dict, num_styles, hidden, dropout, eps = 1e-10):
        super().__init__()
        
        self.eps = eps

        self.sizes = size_dict
        
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        
        # self.fc1_doc = nn.Linear(vocab_size, hidden)
        # self.fc1_meta = nn.Linear(meta_size, hidden)

        self.features = sorted(size_dict.keys())
        self.fc1s = nn.ModuleDict({feature:nn.Linear(self.sizes[feature], hidden) for feature in self.features})
        self.fc2 = nn.Linear(len(self.features) * hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_styles)
        self.fclv = nn.Linear(hidden, num_styles)

        self.bnmu = nn.BatchNorm1d(num_styles, affine=False)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_styles, affine=False)  # to avoid component collapse

    def forward(self, inputs): #inputs_doc, inputs_meta):
        assert isinstance(inputs, dict)
        first_hiddens = []
        for _, feature in enumerate(self.features):
            first_hiddens.append(F.softplus(self.fc1s[feature](inputs[feature])))
        
        h = torch.hstack(first_hiddens)
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs
        logkappa_loc = self.bnmu(self.fcmu(h))
        logkappa_logvar = self.bnlv(self.fclv(h))
        logkappa_scale = self.eps + (0.5 * logkappa_logvar).exp()  # Enforces positivity
        return logkappa_loc, logkappa_scale

class Decoder(nn.Module):
    # Base class for the decoder net, used in the model
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        # the output is σ(βθ)
        return self.bn(self.beta(inputs))
    
class MetaDocDecoder(nn.Module):
    # Base class for the decoder net, used in the model
    def __init__(self, vocab_size, num_topics, num_styles, dropout):
        super().__init__()
        self.t_beta = nn.Linear(num_topics, vocab_size, bias=False)
        self.s_beta = nn.Linear(num_styles, vocab_size, bias=False)
        self.t_bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.s_bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs_doc, inputs_meta):
        inputs_doc  = self.drop(inputs_doc)
        inputs_meta = self.drop(inputs_meta)
        # the output is σ(βθ)
        
        dist_t = self.t_bn(self.t_beta(inputs_doc))
        dist_s = self.s_bn(self.s_beta(inputs_meta))
        
        return 0.5 * (dist_t + dist_s)

class ProdSLDA(nn.Module):
    
    PRIOR_DISTS  = {'gaussian': dist.Normal,
                    'laplace': dist.Laplace,
                   }
    TK_LINKS     = ('none', # Model style and documents independently
                    'kappa_doc', # Allow kappa to effect word distributions
                    'kappa_doc_style', # Allow kappa to effect word distributions and sampled words to effect style
                   )
    
    def __init__(self, vocab_size, meta_sizes, num_topics, num_styles, hidden, dropout, 
                 theta_prior_dist = 'gaussian', theta_prior_loc = 0., theta_prior_scale = 1.,
                 kappa_prior_dist = 'laplace', kappa_prior_loc = 0., kappa_prior_scale = 1.,
                 style_topic_link = 'none',
                 eps = 1e-10):
        super().__init__()
        
        # Global model variables
        self.vocab_size = vocab_size
        self.meta_sizes = meta_sizes
        self.num_topics = num_topics
        self.num_styles = num_styles
        self.hidden     = hidden
        self.dropout    = dropout

        self.meta_features = sorted(self.meta_sizes.keys())
        
        self.eps = eps
        
        # Theta Prior
        if theta_prior_dist not in ProdSLDA.PRIOR_DISTS.keys():
            raise ValueError(f'Theta prior {theta_prior_dist} not yet implemented. Must be one of {", ".join(ProdSLDA.PRIOR_DISTS.keys())}')
        self.theta_prior_dist = theta_prior_dist
        
        self.theta_prior_scale = theta_prior_scale
        self.theta_prior_loc = theta_prior_loc
        
        # Kappa Prior
        if kappa_prior_dist not in ProdSLDA.PRIOR_DISTS.keys():
            raise ValueError(f'Kappa prior {kappa_prior_dist} not yet implemented. Must be one of {", ".join(ProdSLDA.PRIOR_DISTS.keys())}')
        self.kappa_prior_dist = kappa_prior_dist
        
        self.kappa_prior_scale = kappa_prior_scale
        self.kappa_prior_loc = kappa_prior_loc
        
        
        # Document style linking
        self.style_topic_link = style_topic_link
        
        if self.style_topic_link not in ProdSLDA.TK_LINKS:
            raise ValueError(f'Link {self.style_topic_link} not yet implemented. Must be one of {", ".join(ProdSLDA.TK_LINKS)}')
        elif self.style_topic_link == 'none':
            # Independent modeling of style and topic, all normal encoder/decoders
            
            self.encoder = GeneralEncoder({'doc':vocab_size}, num_topics, hidden, dropout, self.eps)
            self.decoder = Decoder(vocab_size, num_topics, dropout)
            self.style_encoder = GeneralEncoder(meta_sizes, num_styles, hidden, dropout, self.eps)
            self.style_decoder = nn.ModuleDict({feature: Decoder(meta_s, num_styles, dropout) for feature, meta_s in meta_sizes.items()})
            
        elif self.style_topic_link == 'kappa_doc':
            # raise NotImplementedError()
            # Doc influences kappa encoding, style encoder takes in doc
            self.encoder = GeneralEncoder({'doc':vocab_size}, num_styles, hidden, dropout, self.eps)
            self.style_encoder = GeneralEncoder({'doc':vocab_size, **meta_sizes}, num_styles, hidden, dropout, self.eps)

            self.decoder = MetaDocDecoder(vocab_size=vocab_size, num_topics=num_topics, num_styles=num_styles, dropout=dropout)
            self.style_decoder = nn.ModuleDict({feature: Decoder(meta_s, num_styles, dropout) for feature, meta_s in meta_sizes.items()})


    def model(self, docs, metas):
        pyro.module("decoder", self.decoder)
        pyro.module("style_decoder", self.style_decoder)
        
        with pyro.plate("documents", docs.shape[0]):
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution
            logtheta_loc = docs.new_zeros((docs.shape[0], self.num_topics)) * self.theta_prior_loc
            logtheta_scale = docs.new_ones((docs.shape[0], self.num_topics)) * self.theta_prior_scale
            logkappa_loc = docs.new_zeros((docs.shape[0], self.num_styles)) * self.kappa_prior_loc
            logkappa_scale = docs.new_ones((docs.shape[0], self.num_styles)) * self.kappa_prior_scale
            
            # if self.style_topic_link == 'kappa_doc':
                # logtheta_s_loc = docs.new_zeros((docs.shape[0], self.num_topics)) * self.theta_prior_loc
                # logtheta_s_scale = docs.new_ones((docs.shape[0], self.num_topics)) * self.theta_prior_scale
                
                # logtheta_s = pyro.sample(
                    # "logtheta_s", ProdSLDA.PRIOR_DISTS[self.theta_prior_dist](logtheta_s_loc, logtheta_s_scale).to_event(1))
                
                # theta_s = F.softmax(logtheta_s, -1)
            
            logtheta = pyro.sample(
                "logtheta", ProdSLDA.PRIOR_DISTS[self.theta_prior_dist](logtheta_loc, logtheta_scale).to_event(1))
            logkappa = pyro.sample(
                "logkappa", ProdSLDA.PRIOR_DISTS[self.kappa_prior_dist](logkappa_loc, logkappa_scale).to_event(1))
            
            theta = F.softmax(logtheta, -1)
            kappa = F.softmax(logkappa, -1)

            if self.style_topic_link == 'none':
                word_logits = self.decoder(theta)
            elif self.style_topic_link == 'kappa_doc':
                word_logits = self.decoder(theta, kappa)
                
            style_logits = {feature:self.style_decoder[feature](kappa) for feature in self.meta_features}

            total_count = int(docs.sum(-1).max())
            pyro.sample(
                'obs_doc',
                dist.Multinomial(total_count, logits = word_logits),
                obs=docs
            )

            for feature in self.meta_features:
                total_s_count = int(metas[feature].sum(-1).max())
                pyro.sample(
                    'obs_meta_'+feature,
                    dist.Multinomial(total_s_count, logits = style_logits[feature]),
                    obs=metas[feature]
                )

    def guide(self, docs, metas):
        pyro.module("encoder", self.encoder)
        pyro.module("style_encoder", self.style_encoder)
            
        with pyro.plate("documents", docs.shape[0]):
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution,
            # where μ and Σ are the encoder network outputs
            
            if self.style_topic_link == 'none':
                logtheta_loc, logtheta_scale = self.encoder({'doc':docs})
                logkappa_loc, logkappa_scale = self.style_encoder(metas)

            elif self.style_topic_link == 'kappa_doc':
                # raise NotImplementedError()
                # logtheta_loc, logtheta_scale, logkappa_d_loc, logkappa_d_scale = self.encoder({'doc':docs, **metas})
                logtheta_loc, logtheta_scale  = self.encoder({'doc':docs})
                logkappa_loc, logkappa_scale = self.style_encoder({'doc':docs, **metas})

                # NICK what was the point of d_loc and d_scale? Shoudln't we be generating one set of kappas from both features?
                
                # Average theta loc from document and style
                # logkappa_loc = 0.5 * (logkappa_loc + logkappa_d_loc) 
                # logkappa_scale = 0.5 * (logkappa_scale + logkappa_d_scale)
            
            # Sample logtheta/logkappa from guide
            logtheta = pyro.sample(
                "logtheta", ProdSLDA.PRIOR_DISTS[self.theta_prior_dist](logtheta_loc, logtheta_scale).to_event(1))
            logkappa = pyro.sample(
                "logkappa", ProdSLDA.PRIOR_DISTS[self.kappa_prior_dist](logkappa_loc, logkappa_scale).to_event(1))

        return logtheta, logkappa
            

    def beta_document(self):
        if self.style_topic_link == 'none':
            return {'beta_topic':self.decoder.beta.weight.cpu().detach().T}
        elif self.style_topic_link == 'kappa_doc':
            return {
                'beta_topic':self.decoder.t_beta.weight.cpu().detach().T,
                'beta_style':self.decoder.s_beta.weight.cpu().detach().T,
            }
        else:
            raise NotImplementedError()
            
    def beta_meta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        # return self.decoder.beta.weight.cpu().detach().T
        retval = {}
        for key, layer in self.style_decoder.items():
            retval[key] = layer.beta.weight.cpu().detach().T
        return retval
            
            