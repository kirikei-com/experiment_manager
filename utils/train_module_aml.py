import pathlib
import subprocess
import time
import datetime

import numpy as np
import pandas as pd

import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
from mxnet.gluon.contrib.estimator import estimator
from mxnet.gluon.contrib.estimator.event_handler import (TrainEnd, EpochEnd, 
                                                         CheckpointHandler)


def mlp(layers, bn=False, act='relu'):
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Flatten())
        for l in layers:
            net.add(nn.Dense(l, activation=act))
            if bn:
                net.add(nn.BatchNorm())
        net.add(nn.Dense(10))
    return net
        
def data_preprocess(setting):
    
    fashion_mnist_train = gluon.data.vision.FashionMNIST(train=True)
    fashion_mnist_val = gluon.data.vision.FashionMNIST(train=False)
    
    transforms = [gluon.data.vision.transforms.Resize(64), 
                gluon.data.vision.transforms.ToTensor()]

    transforms = gluon.data.vision.transforms.Compose(transforms)
    
    fashion_mnist_train = fashion_mnist_train.transform_first(transforms)
    fashion_mnist_val = fashion_mnist_val.transform_first(transforms)
    
    batch_size = setting['batch_size']

    train_data_loader = gluon.data.DataLoader(fashion_mnist_train, batch_size=batch_size, 
                                              shuffle=True, num_workers=4)
    val_data_loader = gluon.data.DataLoader(fashion_mnist_val, batch_size=batch_size, 
                                            shuffle=False, num_workers=4)
    
    return train_data_loader, val_data_loader

def train(train_dl, test_dl, exp, setting, tags=[]):

    num_epochs = setting['epochs']
    opt = setting['opt']
    
    # gpu setting
    gpu_count = setting['gpu_count']
    ctx = [mx.gpu(i) for i in range(gpu_count)] if gpu_count > 0 else mx.cpu()
    
    net = mlp(**setting['model_params'])
    
    net.initialize(init=mx.init.Xavier(), ctx=ctx, force_reinit=True)
    net.hybridize(static_alloc=True, static_shape=True)
    
    trainer = gluon.Trainer(net.collect_params(), 
                            opt, setting['opt_params'])
    
    # metrics
    train_acc = mx.metric.Accuracy()
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    
    est = estimator.Estimator(net=net, 
                              loss=loss_fn, 
                              metrics=train_acc, 
                              trainer=trainer, 
                              context=ctx)
    
    # loggingの開始
    run = exp.start_logging()
    
    try:
        # tagをつける
        for t in tags:
            run.tag(t)
            
        # settingを保存
        log_dict(run, setting)

        # モデルを保存するcallback
        # クラウド上に保存するので/tmpでOK
        checkpoint_handler = CheckpointHandler(model_dir='/tmp',
                                           model_prefix='model',
                                           monitor=train_acc,
                                           save_best=True,
                                           max_checkpoints=0)
        
        # runを利用してAML上にlogging
        record_handler = AMLRecordHandler(run)

        # ignore warnings for nightly test on CI only
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            est.fit(train_data=train_dl, val_data=test_dl,
                epochs=num_epochs, event_handlers=[checkpoint_handler, record_handler])
        
        
        # モデルをアップロード
        run.upload_file(name='model-best.params', path_or_stream = '/tmp/model-best.params')
        
        # statusをcompleteにする
        run.complete()
        
    except Exception as e:
        # statusをfailにする
        run.fail(e)
        raise ValueError('error occured: {}'.format(e))
        

def log_dict(run, params):
    """dict形式の変数をパラメータとして上げる
    """
    for k,v in params.items():
        if isinstance(v, dict):
            log_dict(run, v)
        else:
            run.log(k, str(v))
    
    
    
        
class AMLRecordHandler(TrainEnd, EpochEnd):
    """loss, metricの値をエポックごとに記録しpickleファイルとして吐き出すHandler
    
        pickle file: 
            {'train loss': [0.9, 0.8, ...], 'train acc': [...], ...}
    """
    def __init__(self, run):
        super(AMLRecordHandler, self).__init__()
        self.run = run
        
    def epoch_end(self, estimator, *args, **kwargs):
        for metric in estimator.train_metrics:
            name, val = metric.get()
            # 記録
            self.run.log(name, val)
        
        for metric in estimator.val_metrics:
            name, val = metric.get()
            # 記録
            self.run.log(name, val)
            