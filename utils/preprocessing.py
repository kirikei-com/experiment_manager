import numpy as np
import pandas as pd

import mxnet as mx
from mxnet import nd, gluon, autograd

def data_preprocessing(exp, setting, tags=[]):
    
    run = exp.start_logging(tags)
    
    fashion_mnist_train = gluon.data.vision.FashionMNIST(train=True)
    fashion_mnist_test = gluon.data.vision.FashionMNIST(train=False)
    
    transforms = [gluon.data.vision.transforms.Resize(setting['resize']), 
                gluon.data.vision.transforms.ToTensor()]

    transforms = gluon.data.vision.transforms.Compose(transforms)
    
    fashion_mnist_train = fashion_mnist_train.transform_first(transforms)
    fashion_mnist_test = fashion_mnist_test.transform_first(transforms)
    
    pd.to_pickle((fashion_mnist_train, fashion_mnist_test), 
                 run.path.joinpath('normalized_data.pkl'))
    
    setting['commit_hash'] = run.git_commit(prefix='prep')
    run.save(setting, 'setting.json', mode='json')