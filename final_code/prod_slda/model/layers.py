import torch
import torch.nn as nn
import torch.nn.functional as F



class GeneralEncoder(nn.Module):
    '''
        General encoder for both topics and styles given data
    '''
    
    def __init__(self, size_dict, num_topics, hidden, dropout, eps = 1e-10):
        super().__init__()
        
        self.eps = eps

        self.sizes = size_dict
        
        self.drop = nn.Dropout(dropout)  # to avoid component collapse

        self.features = sorted(size_dict.keys())
        self.fc1s = nn.ModuleDict({feature:nn.Linear(self.sizes[feature], hidden) for feature in self.features})
        self.fc2 = nn.Linear(len(self.features) * hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        
        # Batch norms help to avoid component collapse
        self.bnmu = nn.BatchNorm1d(num_topics, affine=False) 
        self.bnlv = nn.BatchNorm1d(num_topics, affine=False)

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
    '''
        Simple decoder to generate documents/metadata from given topics/styles alone
    '''
    
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        
        return self.bn(self.beta(inputs))
    
class MetaDocDecoder(nn.Module):
    '''
        Decoder generating documents given both topics and styles
    '''
        
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
        dist_t = self.t_bn(self.t_beta(inputs_doc))
        
        if inputs_meta is None:
            return dist_t
        else:
            inputs_meta = self.drop(inputs_meta)
            dist_s = self.s_bn(self.s_beta(inputs_meta))
            
            return 0.5 * (dist_t + dist_s)