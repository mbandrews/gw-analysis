from torch.utils.data import Dataset
import pyarrow.parquet as pq
import numpy as np

def detidx2chnl(detidx):
    if detidx is None:
        return 'H1-L1'
    elif detidx == 0:
        return 'H1'
    elif detidx == 1:
        return 'L1'
    else:
        raise Exception('Unknown det idx:',detidx)

def num2str(num):
    #s = str('%3.2E'%num)
    s = str('%.E'%num)
    s = s.replace('-','m')
    s = s.replace('+','p')
    return s

def clean_str(s):
    print(s)
    print(type(s))
    return s.replace('.','p')

class ParquetDataset(Dataset):
    def __init__(self, filename, scale=2., shift=False, chnl=None, blind=None, ndet=2):
        self.parquet = pq.ParquetFile(filename)
        #self.cols = ['strain','y']
        #self.cols = ['strain', 'y', 'dist', 'm1', 'm2']
        self.cols = ['strain', 'y', 'dist', 'm1', 'm2', 'dec', 'ra']
        self.scale = scale
        self.shift = shift
        self.chnl = chnl
        self.blind = blind
        self.ndet = ndet
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols).to_pydict()
        #data['strain'] = np.float32(np.ones(40960)).reshape(1,-1) #np.float32(data['strain'])*1.5
        #data['strain'] = np.float32(data['strain'])/self.scale
        if self.ndet == 2:
            data['strain'] = np.float32(data['strain'][0]) #2det
            if self.chnl is not None:
                data['strain'] = data['strain'][self.chnl:self.chnl+1] #2det, 0:H1, 1:L1
            if self.blind is not None:
                data['strain'][self.blind] = 0. #2det, 0:H1, 1:L1
        else:
            data['strain'] = np.float32(data['strain']) #1det
        if self.shift:
            data['strain'] -= data['strain'].mean()
        data['strain'] /= self.scale
        data['y'] = np.float32(data['y'])
        #data['y'] = np.int64(data['y'][0])
        data['dist'] = np.float32(data['dist'])
        data['m1'] = np.float32(data['m1'])
        data['m2'] = np.float32(data['m2'])
        data['dec'] = np.float32(data['dec'])
        data['ra'] = np.float32(data['ra'])
        return dict(data)
    def __len__(self):
        return self.parquet.num_row_groups
