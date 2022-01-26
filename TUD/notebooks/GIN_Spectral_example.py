#@title select the dataset { form-width: "30%" }
import argparse
parser = argparse.ArgumentParser(description='graph embedding karate')
parser.add_argument('-d', "--dataset", dest='dataname', metavar='<datset>', type=str, help='dataset name (default MUTAG)',  default='MUTAG', required=False)
parser.add_argument('-V', "--verbose", action='store_true', required=False)

args = parser.parse_args()
dataname = args.dataname

import shutil
import os
shutil.unpack_archive(f'../datasets/{dataname}.zip', '.')
import sys
sys.path.append('..')
from wrappers.spektral_wrapper import MyTUDataset
dataset = MyTUDataset(dataname, path=dataname, force_degree=True)

"""
This example shows how to perform graph classification with a simple Graph
Isomorphism Network.
"""

import numpy as np
import tensorflow as tf
import tqdm.notebook as tq
from time import time
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader
from spektral.datasets import TUDataset
from spektral.layers import GINConv, GlobalSumPool, GlobalAvgPool

################################################################################
# Config
################################################################################
#@title Parameters { form-width: "30%" }
learning_rate = 0.001  #@param {type:"number"}
epochs = 100 #@param {type:"slider", min:0, max:500, step:20}
channels = 64 #@param {type:"slider", min:16, max:128, step:16}
batch_size = 10  #@param {type:"slider", min:1, max:64, step:1}
layers = 3  #@param {type:"slider", min:1, max:5, step:1}
folds = 5  #@param {type:"slider", min:1, max:10, step:1}
verbose = args.verbose #@param {type:"boolean"}
seed = 42 #@param {type:"number"}

# Parameters
F = dataset.n_node_features  # Dimension of node features
n_out = dataset.n_labels  # Dimension of the target
print("Node features", F)
print(dataset[0].x)
#sys.exit(0)
################################################################################
# Build model
################################################################################
class GIN0(Model):
    def __init__(self, channels, n_layers):
        super().__init__()
        self.conv1 = GINConv(channels, epsilon=0, mlp_hidden=[channels, channels])
        self.convs = []
        for _ in range(1, n_layers):
            self.convs.append(
                GINConv(channels, epsilon=0, mlp_hidden=[channels, channels])
            )
        self.pool = GlobalAvgPool()
        self.dense1 = Dense(channels, activation="relu")
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(n_out, activation="softmax")

    def call(self, inputs):
        x, a, = inputs[0:2]
        i = inputs[-1]
        #x, a, i = inputs
        x = self.conv1([x, a])
        for conv in self.convs:
            x = conv([x, a])
        x = self.pool([x, i])
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)


start = time()

# Cross Validation loop
from sklearn.model_selection import StratifiedKFold
targets = [g.y.dot(1 << np.arange(g.y.size)[::-1]) for g in dataset]
sp = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
results = []
for idx_tr, idx_te in tq.tqdm(list(sp.split(dataset, targets)), desc="fold: "):
  dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]

  loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
  loader_te = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1)

################################################################################
# Fit model
################################################################################
  @tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
  def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(categorical_accuracy(target, predictions))
    return loss, acc
  # Build model
  model = GIN0(channels, layers)
  optimizer = Adam(learning_rate)
  loss_fn = CategoricalCrossentropy()

  epoch = step = 0
  tresults = []
  for batch in loader_tr:
    step += 1
    loss, acc = train_step(*batch)
    tresults.append((loss, acc))
    if step == loader_tr.steps_per_epoch:
        step = 0
        epoch += 1
        if verbose: print("Ep. {} - Loss: {}. Acc: {}".format(epoch, *np.mean(tresults, 0)))
        tresults = []

################################################################################
# Evaluate model
################################################################################
  for batch in loader_te:
    inputs, target = batch
    predictions = model(inputs, training=False)
    results.append(
        (
            loss_fn(target, predictions),
            tf.reduce_mean(categorical_accuracy(target, predictions)),
        )
    )
  if verbose: print("Done. Test loss: {}. Test acc: {}".format(*np.mean(results, 0)))
# Timing
temp = time() - start
hours = temp//3600
temp = temp - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
expired = '%d:%d:%d' %(hours,minutes,seconds)
print("Done. Test loss: {}. Test acc: {}".format(*np.mean(results, 0)))

method = 'GIN'
from datetime import datetime
import pandas as pd
path = f'{method}_results.csv'
if not os.path.exists(path):
  dfres = pd.DataFrame(columns=['dataset', 'loss', 'acc', 'folds', 'seed', 'lr', 'epochs', 'batch_size', 'layers', 'channels', 'date', 'elapsed'])
  dfres.to_csv(path, index=False)
dfres = pd.read_csv(path)
dfres = dfres.append({'dataset': dataname,
                      'loss': np.mean(results, 0)[0],
                      'acc' : np.mean(results, 0)[1], 
                      'folds' : folds,
                      'seed' : seed, 
                      'lr' : learning_rate, 'epochs' : epochs, 'batch_size' : batch_size, 'layers': layers, 'channels' : channels, 
                      'date': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                      'elapsed': expired
                      }, ignore_index=True)
dfres.to_csv(path, index=False)
dfres
