import numpy as np # linear algebra
run = 0
np.random.seed(run)
import os, glob
import time
import pyarrow as pa
import pyarrow.parquet as pq
import torch
torch.manual_seed(run)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import matplotlib.pyplot as plt
#plt.switch_backend('agg')
from skimage.transform import rescale
#plt.rcParams["figure.figsize"] = (5,5)
from sklearn import metrics
from torch.utils.data import ConcatDataset, sampler, DataLoader #*
from eval_utils import logger, do_eval_binary
from data_utils import * #ParquetDataset, detidx2str
from network_utils import *

import argparse
parser = argparse.ArgumentParser(description='Run BBH training script.')
parser.add_argument('-e', '--epochs', default=50, type=int, help='Number of training epochs.')
parser.add_argument('-l', '--lr_init', default=5.e-4, type=float, help='Initial learning rate.')
parser.add_argument('-s', '--scale', default=2., type=float, help='Amplitdue preprocessing scale. With BP+whitening: 2.')
parser.add_argument('-t', '--shift', default=False, type=bool, help='Amplitdue preprocessing shift to mean=0. Only needed for wht, noBP.')
parser.add_argument('-b', '--nblocks', default=3, type=int, help='Number of residual blocks.')
parser.add_argument('-c', '--cuda', default=1, type=int, help='Which gpuid to use.')
parser.add_argument('-k', '--kernel', default=7, type=int, help='Resnet conv kernel size.')
parser.add_argument('-d', '--down', default=3, type=int, help='Resnet downsampling conv stride.')
parser.add_argument('-p', '--premax', default=None, type=int, help='Amount of max-pooling to apply prior to NN input.')
parser.add_argument('-i', '--detidx', default=None, type=int, help='Det idx. None:H1+L1, 0:H1-only, 1:L1-only')
parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
args = parser.parse_args()

debug = args.debug
#debug = True
lr_init = args.lr_init
batch_size = 32
epochs = args.epochs
gpuid = args.cuda
nblocks = args.nblocks
kernel = args.kernel
down = args.down
#fmaps = [4, 6, 8, 12, 16] # Resnet_deep
fmaps = [4, 6, 8, 12] # Resnet_stride
#fmaps = [4, 8, 16] # Resnet_premax
#fmaps = [8, 16, 32]
#fmaps = [24, 48, 96]
#premax = 64 # 2s->128
#premax = 16# 0.5s->128
premax = args.premax
shift = args.shift # preproc shift
scale = args.scale #2. # preproc scale
#scale = 20. # noBP preproc scale
#scale = 5.E-20. # nowht, noBP preproc scale
detidx = args.detidx
n_train = 72000
#scenario = 'm10to50_d5E2to1E3_2s'
#scenario = 'm10to50_d5E2to1E3_2s'
scenario = 'm20to50_d1E2to5E2_teff0.5s_tw4.5s_ts0.5s'
#scenario = 'm20to50_d1E2to5E2_teff0.5s_tw4.5s_ts0.5s_nobp'
#scenario = 'm20to50_d1E2to5E2_teff2.0s_tw6.0s_ts1.0s'
#scenario = 'm20to50_d1E2to5E2_teff2.0s_tw6.0s_ts1.0s_nobp'
if debug:
    epochs = 1

###################################################################
# Logging
#run_logger = False
run_logger = True

# Specify which gpuid to use
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpuid)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # :0 is always first *visible* device
#device = torch.device("cpu")

#def num2str(num):
#    #s = str('%3.2E'%num)
#    s = str('%.E'%num)
#    s = s.replace('-','m')
#    s = s.replace('+','p')
#    return s
#
#def clean_str(s):
#    print(s)
#    print(type(s))
#    return s.replace('.','p')

#expt_name = 'BBH-H1-%s_resnet-scale%2.1f-premax%s-fmaps%s-blocks%d_epochs%d-ntrain%.fk-lr%s_run%d'\
#expt_name = 'BBH-H1-%s_resnet-scale%2.1f-shift%s-premax%s-fmaps%s-blocks%d-kernel%d-down%d_epochs%d-ntrain%.fk-lr%s_run%d'\
expt_name = 'BBH-%s-%s_resnet-scale%2.1f-shift%s-premax%s-fmaps%s-blocks%d-kernel%d-down%d_epochs%d-ntrain%.fk-lr%s_run%d'\
        %(detidx2chnl(detidx), scenario, scale, str(shift), str(premax), '-'.join([str(m) for m in fmaps]), nblocks, kernel, down, epochs, n_train//1e3, num2str(lr_init), run)

if debug:
    expt_name = 'DEBUG_'+expt_name

if run_logger:
    if not os.path.isdir('LOGS'):
        os.makedirs('LOGS')
    f = open('LOGS/%s.log'%(expt_name), 'w')
    #for d in ['MODELS', 'METRICS','PLOTS']:
    for d in ['MODELS', 'PLOTS']:
        if not os.path.isdir('%s/%s'%(d, expt_name)):
            os.makedirs('%s/%s'%(d, expt_name))
# else: os.open('/dev/null')

logger(f, '>> Experiment: %s'%(expt_name))
if debug:
    logger(f, '.. running in debug mode !!')

###################################################################
# Load data
# Get parquet files
#pqfiles = glob.glob('data/O1/parquet/%s/*.parquet'%scenario)
#pqfiles = glob.glob('data/O1/parquet/%s/O1_H1_*.parquet'%scenario)
pqfiles = glob.glob('data/O1/parquet/%s/O1_H1-L1_*.parquet'%scenario)
logger(f, '>> Input files: %s'%(' '.join(pqfiles)))
dset_train = ConcatDataset([ParquetDataset(pqf, scale, shift, detidx) for pqf in pqfiles])

# Train set _______________________________________________________________________
# Take subset of training set, if desired
idxs = np.random.permutation(len(dset_train))
logger(f, '>> Random(10): %s'%(' '.join([str(n) for n in np.random.permutation(10)])))
#print(len(idxs))
if n_train != -1:
    assert n_train <= len(idxs)
    idxs_train = idxs[:n_train]
    idxs_val = idxs[n_train:]

# Create data loader
train_sampler = sampler.SubsetRandomSampler(idxs_train)
train_loader = DataLoader(dataset=dset_train, batch_size=batch_size, num_workers=2, pin_memory=True, sampler=train_sampler)

# Test set _______________________________________________________________________
val_sampler = sampler.SubsetRandomSampler(idxs_val)
val_loader = DataLoader(dataset=dset_train, batch_size=batch_size, num_workers=2, sampler=val_sampler)

logger(f, '>> N samples, train: %d'%(len(idxs_train)))
logger(f, '>> N samples, test: %d'%(len(idxs_val)))

###################################################################
# Network architecture
#model = ResNet(in_channels=1, nblocks=nblocks, fmaps=fmaps, kernel=3, debug=debug)
in_channels = 2 if detidx is None else 1
if premax is None:
    if len(fmaps) == 5:
        logger(f, '>> Model: Resnet_deep')
        model = ResNet_deep(in_channels=in_channels, nblocks=nblocks, fmaps=fmaps, kernel=kernel, debug=debug)
    else:
        logger(f, '>> Model: Resnet_stride')
        model = ResNet_stride(in_channels=in_channels, nblocks=nblocks, fmaps=fmaps, down=down, kernel=kernel, debug=debug)
else:
    logger(f, '>> Model: Resnet_premax')
    model = ResNet_premax(in_channels=in_channels, nblocks=nblocks, fmaps=fmaps, premax=premax, kernel=kernel, debug=debug)
model.to(device)#.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr_init)

###################################################################
# Main training loop

logger(f, ">> Training <<<<<<<<")
print_step = len(train_loader)//10 if len(train_loader) > 5 else 1
aucs = []
tpr_fpr0s = []
losses = []

for e in range(epochs):

    epoch = e+1
    ntrained = 0
    logger(f, '>> Epoch %d <<<<<<<<'%(epoch))

    # Run training
    model.train()
    now = time.time()
    for i, data in enumerate(train_loader):
        #print(len(data))
        #X, y_true_ = data
        #X, y_true = data[0].to(device), data[1].to(device)
        X, y_true = data['strain'].to(device), data['y'].to(device)
        # Check for NaNs:
        #nan_eles = torch.isnan(X).sum(2).view(-1)
        nan_eles = torch.isnan(X).sum(2).sum(1)
        #if nan_eles.sum() > 0:
        #    break
        nan_rows = (nan_eles > 1)
        X, y_true = X[~nan_rows], y_true[~nan_rows]
        # Downsample waveform prior to CNN
        #X = F.max_pool1d(X, kernel_size=64)#/3.
        #print(X.size(), y_true.size())
        #print('ytrue:', y_true.size(), y_true)
        #break
        optimizer.zero_grad()
        logits = model(X).to(device)
        #break

        #print(logits.size())
        #print('ypred:',y_pred.size(),y_pred)
        loss = F.binary_cross_entropy_with_logits(logits, y_true).to(device)
        #losses.append(loss.item())
        #print('loss',loss)
        #break
        loss.backward()
        optimizer.step()
        ntrained += len(logits)
        if i % print_step == 0:
            pass
            #y_pred = torch.argmax(logits, dim=1)
            y_pred = logits.ge(0.).byte()
            #print('ypred:',y_pred.size(),y_pred)
            #y_pred = F.softmax(logits, 1) # only if want output normalized to (0,1)
            #y_pred = torch.argmax(y_pred, dim=1)
            #print('ypred,softmax:',y_pred.size(),y_pred)
            acc = metrics.accuracy_score(y_true=y_true.tolist(), y_pred=y_pred.tolist())
            logger(f, '%d: (%d/%d) y_true: %s...'%(epoch, i, len(train_loader), str(np.squeeze(y_true.tolist()[:5]))))
            logger(f, '%d: (%d/%d) y_pred: %s...'%(epoch, i, len(train_loader), str(np.squeeze(y_pred.tolist()[:5]))))
            logger(f, '%d: (%d/%d) train loss:%f, acc:%f'%(epoch, i, len(train_loader), loss.item(), acc))
        #break
        if torch.isnan(loss):
            break
        if debug:
            Xnp = np.concatenate(X.cpu().numpy()).flatten()
            print('strain, max:%f, stdev:%f'%(Xnp.max(), np.std(Xnp)))
            break
    now = time.time() - now
    #y_pred = torch.argmax(logits, dim=1)
    y_pred = logits.ge(0.).byte()
    acc = metrics.accuracy_score(y_true=y_true.tolist(), y_pred=y_pred.tolist())
    logger(f, '%d: Train time:%.2fs in %d steps for N:%d samples'%(epoch, now, len(train_loader), ntrained))
    logger(f, '%d: Train loss:%f, acc:%f'%(epoch, loss.item(), acc))

    # Run Validation
    model.eval()
    logger(f, '>> Validation <<<<<<<<')
    losses, aucs, tpr_fpr0s, _, _, _, _, _, _, _, _, _ = do_eval_binary(model, optimizer, device, val_loader, expt_name, epoch, losses, aucs, tpr_fpr0s, f, debug)

losses, aucs, tpr_fpr0s = np.array(losses), np.array(aucs), np.array(tpr_fpr0s)
logger(f, '>> Best Loss:%4.3E +/- %4.3E @ epoch:%d'%(np.min(losses), np.min(losses)/np.sqrt(len(idxs_val)), np.argwhere(losses==np.min(losses))[0][0]+1))
logger(f, '>> Best AUC:%4.3f +/- %4.3f @ epoch:%d'%(np.max(aucs), np.max(aucs)/np.sqrt(len(idxs_val)), np.argwhere(aucs==np.max(aucs))[0][0]+1))
logger(f, '>> Best TPR@FPR=0:%4.3f +/- %4.3f @ epoch:%d'%(np.max(tpr_fpr0s), np.max(tpr_fpr0s)/np.sqrt(len(idxs_val)*np.max(tpr_fpr0s)/2.), np.argwhere(tpr_fpr0s==np.max(tpr_fpr0s))[0][0]+1))

if run_logger:
    pass
    f.close()
