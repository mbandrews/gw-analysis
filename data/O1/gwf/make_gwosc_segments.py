import numpy as np
import os
import json

idx = 0
idxmax = 0
fh = np.array(open('O1_H1_gwosc_list.txt').readlines())
fl = np.array(open('O1_L1_gwosc_list.txt').readlines())
hidxs = np.array([f.strip().split('-')[-2] for f in fh])
lidxs = np.array([f.strip().split('-')[-2] for f in fl])
print('idx:',idx)
outdir = 'segments'
if not os.path.isdir(outdir):
    os.makedirs(outdir)
hout = open('%s/O1_H1_gwosc_idx%03d_list.txt'%(outdir, idx), 'w')
lout = open('%s/O1_L1_gwosc_idx%03d_list.txt'%(outdir, idx), 'w')
ncommon = 0
isdead = False
for i in range(len(hidxs)):

    #if hidxs[i] in lidxs:
    if i < len(hidxs)-1:
        diff = int(hidxs[i+1]) - int(hidxs[i])
        iscont = True if diff == 4096 else False
    else:
        iscont = True

    #if hidxs[i] in lidxs:
    if hidxs[i] in lidxs and iscont:
        h2lidx = (lidxs == hidxs[i])
        #print(h2lidx)
        #print(h2lidx.shape)
        #print(idx, i, hidxs[i], lidxs[h2lidx][0])
        hout.write(fh[i])
        lout.write(fl[h2lidx][0])
        isdead = False
        ncommon += 1
    else:
        if not isdead:
            idx += 1
            hout.close()
            lout.close()
            if idx > idxmax: break
            hout = open('%s/O1_H1_gwosc_idx%03d_list.txt'%(outdir, idx), 'w')
            lout = open('%s/O1_L1_gwosc_idx%03d_list.txt'%(outdir, idx), 'w')
        isdead = True
        #print('idx:',idx)
    #print(fh[i].strip())
    #fidx = fh[i].strip().split('-')[-2]
    #print(fidx)
    #lidx = fl[i].strip().split('-')[-2]
    #print(lidx)
    #fout.write(u+'\n')
    #break
    #if i > 60: break
#fout.close()
hout.close()
print(ncommon, idx)
