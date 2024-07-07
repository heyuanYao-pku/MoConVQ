import numpy as np
import yaml
import torch

def add_to_list(src, tar):
    if isinstance(src, list):
        src = np.concatenate(src, axis = 0)
            
    if tar is None:
        return src
    else:
        return np.concatenate([tar,src], axis = 0)

def load_data(path = None, fdargs = {}, loadargs = {}):
    if path is None:
        import tkinter.filedialog as fd
        path = fd.askopenfilename(**fdargs)
    with open(path,'rb') as f:
        data = torch.load(f, **loadargs)
    return data


def flatten_dict(dd, separator='_', prefix=''):
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }
    
def load_yaml(path = None, **kargs):
    if path is None:
        import tkinter.filedialog as fd
        path = fd.askopenfilename(filetypes=[('YAML','*.yml')], **kargs)
    with open(path,'r') as f:
        config = yaml.safe_load(f)
    return flatten_dict(config)