import os
import numpy as np

idx = 2
print('>> segment batch idx:',idx)

indir = 'segments'
hin = open('%s/O1_H1_gwosc_idx%03d_list.txt'%(indir, idx)).readlines()
lin = open('%s/O1_L1_gwosc_idx%03d_list.txt'%(indir, idx)).readlines()
hfiles = np.array([f.strip().split('/')[-1] for f in hin])
lfiles = np.array([f.strip().split('/')[-1] for f in lin])

#outdir = 'segments'
outdir = '.'
if not os.path.isdir(outdir):
    os.makedirs(outdir)
hout = open('%s/O1_H1_local_idx%03d_list.txt'%(outdir, idx), 'w')
lout = open('%s/O1_L1_local_idx%03d_list.txt'%(outdir, idx), 'w')

datadir = '%s/segments'%os.getcwd()
for hf, lf in zip(hfiles, lfiles):
    print(hf, lf)
    hf = '%s/%s'%(datadir, hf)
    lf = '%s/%s'%(datadir, lf)
    assert os.path.isfile(hf)
    assert os.path.isfile(lf)
    hout.write(hf+'\n')
    lout.write(lf+'\n')
#
#for u in urls[:nfiles_cut]:
#    print(u)
#    fout.write(u+'\n')
#    #os.system('wget -q %s'%u)
#fout.close()
#
#
#gwf_data = []
## Metadata is stored in `strain`
#for d in data['strain']:
#
#    # json file contains both hdf5 and gwf metadata
#    # pick only gwfs
#    if d['format'] != 'gwf': continue
#
#    # Copy all the keys
#    gwf_data_ = {}
#    for k in d.keys():
#        gwf_data_[k] = d[k]
#
#    # Keep a list over all files
#    gwf_data.append(gwf_data_)
#
#f.close()
#
## Get just the urls
#urls = [u['url'] for u in gwf_data]
#print('>> N files:',len(urls))
#
## Skip first file since has lots of NaNs
#print('.. excluding first file:',urls[0])
#urls = urls[1:]
#
##print('.. will download first %d files:'%nfiles_cut)
#print('.. will write first %d files:'%nfiles_cut)
##fout = open('O1_%d_local_idx%04d_list.txt'%(det, segidx), 'w+')
#fout = open('test_%d_local_idx%04d_list.txt'%(det, segidx), 'w+')
#for u in urls[:nfiles_cut]:
#    print(u)
#    fout.write(u+'\n')
#    #os.system('wget -q %s'%u)
#fout.close()
#
