import os
import json

# N files to output to list
#nfiles_cut = 2
nfiles_cut = 50

# Download json file from here: https://www.gw-openscience.org/archive/O1/
# For example, for L1, will bring you to the ff url: https://www.gw-openscience.org/archive/links/O1/L1/1126051217/1137254417/json/
# then do
# $ wget https://www.gw-openscience.org/archive/links/O1/L1/1126051217/1137254417/json/ .
# $ mv index.html O1_L1_gwosc.json
#json_file = 'O1_H1_gwosc.json'
json_file = 'O1_L1_gwosc.json'
f = open(json_file)
data = json.load(f)

gwf_data = []
# Metadata is stored in `strain`
for d in data['strain']:

    # json file contains both hdf5 and gwf metadata
    # pick only gwfs
    if d['format'] != 'gwf': continue

    # Copy all the keys
    gwf_data_ = {}
    for k in d.keys():
        gwf_data_[k] = d[k]

    # Keep a list over all files
    gwf_data.append(gwf_data_)

f.close()

# Get just the urls
urls = [u['url'] for u in gwf_data]
print('>> N files:',len(urls))

# Skip first file since has lots of NaNs
if 'H1' in json_file:
    print('.. excluding first file:',urls[0])
    urls = urls[1:]
else:
    print('.. excluding first 2 files:',urls[:2])
    urls = urls[2:]

idx = 0
#print('.. will download first %d files:'%nfiles_cut)
print('.. will write first %d files:'%nfiles_cut)
#fout = open('O1_%s_gwosc_N%03d_index%04d_list.txt'%(json_file.split('_')[1], nfiles_cut, idx), 'w+')
fout = open('O1_%s_gwosc_list.txt'%(json_file.split('_')[1]), 'w+')
#for u in urls[idx*nfiles_cut:(idx+1)*nfiles_cut]:
for u in urls:
    print(u)
    fout.write(u+'\n')
    #os.system('wget -q %s'%u)
fout.close()

