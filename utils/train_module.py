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

def train(exp, setting, tags=[]):

    # データの取得
    # 本来は実行時にnormalizeなどの変換をしてもいいが説明のために分けている
    fashion_mnist_train, fashion_mnist_test = pd.read_pickle(setting['data_path'])
    
    batch_size = setting['batch_size']
    train_dl = gluon.data.DataLoader(fashion_mnist_train, batch_size=batch_size, 
                                              shuffle=True, num_workers=4)
    test_dl = gluon.data.DataLoader(fashion_mnist_test, batch_size=batch_size, 
                                            shuffle=False, num_workers=4)
    
    
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
    
    run = exp.start_logging(tags)
    
    try:
        setting['commit_hash'] = run.git_commit()
        run.save(setting, 'setting.json', mode='json')

        checkpoint_handler = CheckpointHandler(model_dir=str(run.path),
                                           model_prefix='model',
                                           monitor=train_acc,
                                           save_best=True,
                                           max_checkpoints=0)
        
        record_handler = RecordHandler(file_name='log.pkl', 
                                       file_location=run.path)

        # ignore warnings for nightly test on CI only
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            est.fit(train_data=train_dl, val_data=test_dl,
                epochs=num_epochs, event_handlers=[checkpoint_handler, record_handler])
        
    except Exception as e:
        run.delete()
        raise ValueError('error occured and delete run folder: {}'.format(e))
        

class RecordHandler(TrainEnd, EpochEnd):
    """loss, metricの値をエポックごとに記録しpickleファイルとして吐き出すHandler
    
        pickle file: 
            {'train loss': [0.9, 0.8, ...], 'train acc': [...], ...}
    """
    def __init__(self, file_name, file_location):
        super(RecordHandler, self).__init__()
        self.history = {}
        self.file_path = file_location.joinpath(file_name)

    def train_end(self, estimator, *args, **kwargs):
        # Print all the losses at the end of training
        print("Training ended")
        pd.to_pickle(self.history, self.file_path)
        
    def epoch_end(self, estimator, *args, **kwargs):
        for metric in estimator.train_metrics:
            name, val = metric.get()
            self.history.setdefault(name, []).append(val)
        
        for metric in estimator.val_metrics:
            name, val = metric.get()
            self.history.setdefault(name, []).append(val)
            