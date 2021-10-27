import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.rcParams["figure.figsize"] = (6, 4)
from sklearn import metrics

def logger(f, s):
    print(s)
    f.write('%s\n'%str(s))

def do_eval_binary_sky(model, optimizer, device, val_loader, expt_name, epoch, losses, aucs, tpr_fpr0s, f=open('/dev/null', 'w'), debug=False, write_model=True):

    eval_step = len(val_loader)//5 if len(val_loader) > 2 else 1
    loss = 0.
    neval = 0
    y_trues, y_preds, y_probs = [], [], []
    dists, m1s, m2s = [], [], []
    decs, ras = [], []
    nnouts = []

    now = time.time()
    for i, data in enumerate(val_loader):
        X, y_true = data['strain'].to(device), data['y'].to(device)
        # Check for NaNs:
        #nan_eles = torch.isnan(X).sum(2).view(-1)
        nan_eles = torch.isnan(X).sum(2).sum(1)
        nan_rows = (nan_eles > 1)
        X, y_true = X[~nan_rows], y_true[~nan_rows]
        # Reduce to 128 length
        #X = F.max_pool1d(X, kernel_size=64)#/3.
        logits = model(X).to(device)

        loss += F.binary_cross_entropy_with_logits(logits, y_true).item() # loss will only converge to true mean in limit of identical batch sizes
        #loss += F.cross_entropy(logits, y_true).item() # loss will only converge to true mean in limit of identical batch sizes

        y_trues.append(y_true.tolist())
        #y_preds.append(torch.argmax(logits, dim=1).tolist()) # class pred only
        #y_probs.append(F.softmax(logits, dim=1).tolist()) # model output, normalized to (0,1)
        y_preds.append(logits.ge(0.).byte().tolist()) # class pred only
        y_probs.append(torch.sigmoid(logits).tolist()) # model output, normalized to (0,1)
        nnouts.append(logits.tolist())

        dist, m1, m2 = data['dist'][~nan_rows], data['m1'][~nan_rows], data['m2'][~nan_rows]
        dec, ra = data['dec'][~nan_rows], data['ra'][~nan_rows]
        dists.append(dist.tolist())
        m1s.append(m1.tolist())
        m2s.append(m2.tolist())
        decs.append(dec.tolist())
        ras.append(ra.tolist())

        neval += len(logits)
        if i % eval_step == 0:
            pass
            logger(f, '%d: (%d/%d) ...'%(epoch, i, len(val_loader)))

        if debug:
            break
    now = time.time() - now
    y_trues = np.concatenate(y_trues).squeeze()
    y_preds = np.concatenate(y_preds).squeeze()
    y_probs = np.concatenate(y_probs).squeeze()
    #print(y_true.shape, y_pred.shape, y_prob.shape)
    dists = np.concatenate(dists).squeeze()
    m1s = np.concatenate(m1s).squeeze()
    m2s = np.concatenate(m2s).squeeze()
    decs = np.concatenate(decs).squeeze()
    ras = np.concatenate(ras).squeeze()
    nnouts = np.concatenate(nnouts).squeeze()

    #acc = metrics.accuracy_score(y_true=y_trues.tolist(), y_pred=y_preds.tolist())
    fprs, tprs, thrs = metrics.roc_curve(y_trues, y_probs) #y_preds)
    auc = metrics.auc(fprs, tprs)
    idx_fpr0 = np.argwhere(fprs==0.)[-1][0]
    tpr_fpr0 = tprs[idx_fpr0]
    logger(f, '%d: Val time:%.2fs in %d steps for N:%d samples'%(epoch, now, len(val_loader), neval))
    logger(f, '%d: Val loss:%f, auc:%f, TPR@FPR=0:%f'%(epoch, loss/len(val_loader), auc, tpr_fpr0)) # loss will only converge to true mean in limit of identical batch sizes

    #'''
    # Save model
    #if auc > best_auc:
        #best_acc = acc
    if write_model:
        score_str = 'epoch%d_auc%.4f'%(epoch, auc)
        filename = 'MODELS/%s/model_%s.pkl'%(expt_name, score_str)
        model_dict = {'model': model.state_dict(), 'optim': optimizer.state_dict()}
        torch.save(model_dict, filename)
    #'''

    # Keep running metrics
    losses.append(loss/len(val_loader))
    aucs.append(auc)
    tpr_fpr0s.append(tpr_fpr0)

    if write_model:
        # Make plots
        # Running loss
        plt.plot(losses, label='Loss, val')
        plt.xlabel('Epoch', size=16)
        plt.ylabel('Loss, val', size=16)
        plt.legend(loc='upper center')
        plt.savefig('PLOTS/%s/%s_loss.png'%(expt_name, score_str), bbox_inches='tight')
        plt.close()
        # Running AUC
        plt.plot(aucs, label='ROC AUC, val')
        plt.xlabel('Epoch', size=16)
        plt.ylabel('ROC AUC, val', size=16)
        plt.legend(loc='upper center')
        plt.savefig('PLOTS/%s/%s_auc.png'%(expt_name, score_str), bbox_inches='tight')
        plt.close()
        # Running AUC
        plt.plot(tpr_fpr0s, label='TPR@FPR=0, val')
        plt.xlabel('Epoch', size=16)
        plt.ylabel('TPR@FPR=0, val', size=16)
        plt.legend(loc='upper center')
        plt.savefig('PLOTS/%s/%s_fprtpr.png'%(expt_name, score_str), bbox_inches='tight')
        plt.close()

    #return best_acc
    return losses, aucs, tpr_fpr0s, fprs, tprs, thrs, y_trues, y_preds, y_probs, dists, m1s, m2s, decs, ras, nnouts

def do_eval_binary(model, optimizer, device, val_loader, expt_name, epoch, losses, aucs, tpr_fpr0s, f=open('/dev/null', 'w'), debug=False, write_model=True):

    eval_step = len(val_loader)//5 if len(val_loader) > 2 else 1
    loss = 0.
    neval = 0
    y_trues, y_preds, y_probs = [], [], []
    dists, m1s, m2s = [], [], []

    now = time.time()
    for i, data in enumerate(val_loader):
        X, y_true = data['strain'].to(device), data['y'].to(device)
        # Check for NaNs:
        #nan_eles = torch.isnan(X).sum(2).view(-1)
        nan_eles = torch.isnan(X).sum(2).sum(1)
        nan_rows = (nan_eles > 1)
        X, y_true = X[~nan_rows], y_true[~nan_rows]
        # Reduce to 128 length
        #X = F.max_pool1d(X, kernel_size=64)#/3.
        logits = model(X).to(device)

        loss += F.binary_cross_entropy_with_logits(logits, y_true).item() # loss will only converge to true mean in limit of identical batch sizes
        #loss += F.cross_entropy(logits, y_true).item() # loss will only converge to true mean in limit of identical batch sizes

        y_trues.append(y_true.tolist())
        #y_preds.append(torch.argmax(logits, dim=1).tolist()) # class pred only
        #y_probs.append(F.softmax(logits, dim=1).tolist()) # model output, normalized to (0,1)
        y_preds.append(logits.ge(0.).byte().tolist()) # class pred only
        y_probs.append(torch.sigmoid(logits).tolist()) # model output, normalized to (0,1)

        dist, m1, m2 = data['dist'][~nan_rows], data['m1'][~nan_rows], data['m2'][~nan_rows]
        dists.append(dist.tolist())
        m1s.append(m1.tolist())
        m2s.append(m2.tolist())

        neval += len(logits)
        if i % eval_step == 0:
            pass
            logger(f, '%d: (%d/%d) ...'%(epoch, i, len(val_loader)))

        if debug:
            break
    now = time.time() - now
    y_trues = np.concatenate(y_trues).squeeze()
    y_preds = np.concatenate(y_preds).squeeze()
    y_probs = np.concatenate(y_probs).squeeze()
    #print(y_true.shape, y_pred.shape, y_prob.shape)
    dists = np.concatenate(dists).squeeze()
    m1s = np.concatenate(m1s).squeeze()
    m2s = np.concatenate(m2s).squeeze()

    #acc = metrics.accuracy_score(y_true=y_trues.tolist(), y_pred=y_preds.tolist())
    fprs, tprs, thrs = metrics.roc_curve(y_trues, y_probs) #y_preds)
    auc = metrics.auc(fprs, tprs)
    idx_fpr0 = np.argwhere(fprs==0.)[-1][0]
    tpr_fpr0 = tprs[idx_fpr0]
    logger(f, '%d: Val time:%.2fs in %d steps for N:%d samples'%(epoch, now, len(val_loader), neval))
    logger(f, '%d: Val loss:%f, auc:%f, TPR@FPR=0:%f'%(epoch, loss/len(val_loader), auc, tpr_fpr0)) # loss will only converge to true mean in limit of identical batch sizes

    #'''
    # Save model
    #if auc > best_auc:
        #best_acc = acc
    if write_model:
        score_str = 'epoch%d_auc%.4f'%(epoch, auc)
        filename = 'MODELS/%s/model_%s.pkl'%(expt_name, score_str)
        model_dict = {'model': model.state_dict(), 'optim': optimizer.state_dict()}
        torch.save(model_dict, filename)
    #'''

    # Keep running metrics
    losses.append(loss/len(val_loader))
    aucs.append(auc)
    tpr_fpr0s.append(tpr_fpr0)

    if write_model:
        # Make plots
        # Running loss
        plt.plot(losses, label='Loss, val')
        plt.xlabel('Epoch', size=16)
        plt.ylabel('Loss, val', size=16)
        plt.legend(loc='upper center')
        plt.savefig('PLOTS/%s/%s_loss.png'%(expt_name, score_str), bbox_inches='tight')
        plt.close()
        # Running AUC
        plt.plot(aucs, label='ROC AUC, val')
        plt.xlabel('Epoch', size=16)
        plt.ylabel('ROC AUC, val', size=16)
        plt.legend(loc='upper center')
        plt.savefig('PLOTS/%s/%s_auc.png'%(expt_name, score_str), bbox_inches='tight')
        plt.close()
        # Running AUC
        plt.plot(aucs, label='TPR@FPR=0, val')
        plt.xlabel('Epoch', size=16)
        plt.ylabel('TPR@FPR=0, val', size=16)
        plt.legend(loc='upper center')
        plt.savefig('PLOTS/%s/%s_fprtpr.png'%(expt_name, score_str), bbox_inches='tight')
        plt.close()

    #return best_acc
    return losses, aucs, tpr_fpr0s, fprs, tprs, thrs, y_trues, y_preds, y_probs, dists, m1s, m2s
