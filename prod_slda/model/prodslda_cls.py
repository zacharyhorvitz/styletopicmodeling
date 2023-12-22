import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist

from .layers import *

class ProdSLDA(nn.Module):
    
    PRIOR_DISTS  = {'gaussian': dist.Normal,
                    'laplace': dist.Laplace,
                   }
    TK_LINKS     = ('none', # Model style and documents independently
                    'kappa_doc', # Allow kappa to effect word distributions
                   )
    
    def __init__(self, vocab_size, meta_sizes, num_topics, num_styles, 
                 topic_hidden, style_hidden, dropout, full_hidden = None,
                 theta_prior_dist = 'gaussian', theta_prior_loc = 0., theta_prior_scale = 1.,
                 kappa_prior_dist = 'laplace', kappa_prior_loc = 0., kappa_prior_scale = 1.,
                 style_topic_link = 'none',
                 eps = 1e-10):
        super().__init__()
        
        # Global model variables
        self.vocab_size   = vocab_size
        self.meta_sizes   = meta_sizes
        self.num_topics   = num_topics
        self.num_styles   = num_styles
        self.topic_hidden = topic_hidden
        self.style_hidden = style_hidden
        self.dropout      = dropout

        self.meta_features = sorted(self.meta_sizes.keys())
        self.hidden_dict   = {feat: style_hidden for feat in self.meta_features}
        self.hidden_dict['doc'] = topic_hidden
        
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
        
        self.full_hidden = int((self.topic_hidden + self.style_hidden)/2.) if full_hidden is None else full_hidden
        
        if self.style_topic_link not in ProdSLDA.TK_LINKS:
            raise ValueError(f'Link {self.style_topic_link} not yet implemented. Must be one of {", ".join(ProdSLDA.TK_LINKS)}')
        elif self.style_topic_link == 'none':
            # Independent modeling of style and topic, all normal encoder/decoders
            
            self.encoder = GeneralEncoder({'doc':vocab_size}, num_topics, {'doc': self.topic_hidden}, self.full_hidden, dropout, self.eps)
            self.decoder = Decoder(vocab_size, num_topics, dropout)
            self.style_encoder = GeneralEncoder(meta_sizes, num_styles, self.hidden_dict, self.full_hidden, dropout, self.eps)
            self.style_decoder = nn.ModuleDict({feature: Decoder(meta_s, num_styles, dropout) for feature, meta_s in meta_sizes.items()})
            
        elif self.style_topic_link == 'kappa_doc':
            # raise NotImplementedError()
            # Doc influences kappa encoding, style encoder takes in doc
            self.encoder = GeneralEncoder({'doc':vocab_size}, num_topics, {'doc': self.topic_hidden}, self.full_hidden, dropout, self.eps)
            self.style_encoder = GeneralEncoder({'doc':vocab_size, **meta_sizes}, num_styles, self.hidden_dict, self.full_hidden, dropout, self.eps)

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

        # print('logtheta_loc', logtheta_loc)
        # print('logkappa_loc', logtheta_scale)
        return logtheta_loc, logkappa_loc
            

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
    
    def reconstruct_doc(self, inputs, use_style = True):
        
        self.eval()
        
        with torch.no_grad():
            if self.style_topic_link == 'none':
                logtheta_loc, _  = self.encoder({'doc':inputs['bow_h1']})
                theta = F.softmax(logtheta_loc, -1)
                word_logits = self.decoder(theta)

            elif self.style_topic_link == 'kappa_doc':
                logtheta_loc, _  = self.encoder({'doc':inputs['bow_h1']})
                theta = F.softmax(logtheta_loc, -1)

                if use_style:
                    logkappa_loc, _ = self.style_encoder({'doc':inputs['bow_h1'], **inputs['meta']})
                    kappa = F.softmax(logkappa_loc, -1)
                else:
                    kappa = None

                word_logits = self.decoder(theta, kappa)

            recon = F.softmax(word_logits, dim = -1)

            return recon
    
    def reconstruct_style(self, inputs):
        '''
            Reconstruct style/metadata from document alone by zeroing out metadata.
        '''
        
        self.eval()
        
        with torch.no_grad():
        
            zero_meta = {k:torch.zeros_like(inputs['meta'][k]) for k in inputs['meta'].keys()}

            logkappa_loc, _ = self.style_encoder({'doc':inputs['bow'], **zero_meta})

            kappa = F.softmax(logkappa_loc, dim = -1)

            s_recon = {feature:self.style_decoder[feature](kappa) for feature in self.meta_features}
            s_recon = {feature:F.softmax(s_recon[feature], dim = -1) for feature in self.meta_features}

            return s_recon
    
    def doc_reconstruct_ce(self, inputs, use_style = True):
    
        recon = self.reconstruct_doc(inputs, use_style = use_style)

        total_count = inputs['bow_h2'].sum()

        ce = (-inputs['bow_h2']*torch.log(recon)).sum().cpu().item()

        return ce, total_count
    
    def style_reconstruct_ce(self, inputs):
        
        s_recon = ProdSLDA.reconstruct_style(self, inputs)
        
        style_ce, total_s_count = 0, 0
        for feature in self.meta_features:
            total_s_count += inputs['meta'][feature].sum()
            
            style_ce += (-inputs['meta'][feature]*torch.log(s_recon[feature])).sum().cpu().item()
        
        return style_ce, total_s_count
            