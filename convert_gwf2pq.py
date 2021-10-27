import os, glob, time
import numpy as np
from numpy.lib.stride_tricks import as_strided
from proc_utils import *

from gwdatafind import find_urls
#import gwpy as gwpy
from gwpy.io.gwf import get_channel_names
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
#import pycbc as pycbc
from pycbc import frame
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector

import pyarrow.parquet as pq
import pyarrow as pa

#import matplotlib.pyplot as plt
# Not the same module the TimeSeries.Plot() calls, unfortunately
#plt.rcParams['pdf.fonttype'] = 42
#plt.rcParams['mathtext.fontset'] = 'cm'
#%matplotlib inline

import argparse
parser = argparse.ArgumentParser(description='Convert and process gwf strain data to parquet for training.')
#parser.add_argument('-i', '--inputlist', required=True, type=str, help='Input gwf file list.')
parser.add_argument('-i', '--inputlist_h1', required=True, type=str, help='Input gwf file list for H1.')
parser.add_argument('-l', '--inputlist_l1', required=True, type=str, help='Input gwf file list for L1.')
parser.add_argument('-w', '--t_window', default=6., type=float, help='Gross time segment window (sec).')
parser.add_argument('-s', '--t_stride', default=0.5, type=float, help='Time segment stride lengths (sec).')
parser.add_argument('--spin', action='store_true', help='Include BBH spin?')
parser.add_argument('--inj_sg', action='store_true', help='Inject BBH signal?')
parser.add_argument('--nobp', action='store_true', help='Skip analysis bandpass?')
parser.add_argument('--nowht', action='store_true', help='Skip whitening?')
args = parser.parse_args()

# List of detectors to use
kH1, kL1 = 'H1', 'L1'
ks = [kH1, kL1]

inputlist = {}
inputlist[kH1] = args.inputlist_h1
inputlist[kL1] = args.inputlist_l1
inj_sg = args.inj_sg
spin = args.spin
nobp = args.nobp
nowht = args.nowht

###################################################################
# Processing params
t_window   = args.t_window #6 #4.5 #6 #4.5 #6 #16 #12 # s
t_stride   = args.t_stride #0.5 #1 #0.5 #1 #4 #12 # s
#end_offset = t_window - t_stride # s, how much to read into following file, for continuity at file boundaries
end_offset = 0 # s, how much to read into following file, for continuity at file boundaries
fcalib_lo  = 16. # Hz
#fcalib_hi  = 2048. # Hz, NOTE: O1 4KHz data already has a 2 kHz cutoff
fsg_lo     = 35.
fsg_hi     = 250.
if nobp:
    fsg_lo = None
    fsg_hi = None
whiten_fft = (4, 2) # s (fft length, stride) in s
if nowht:
    whiten_fft = None
#inj_sg = True
#inj_sg = False
print('>> Making waveforms...')
print('.. window length: %d s'%t_window)
print('.. stride length: %d s'%t_stride)
print('.. end time offset: %d s'%end_offset)
print('.. highpass preselection: f > %d Hz'%fcalib_lo)
#print('.. bandpass preselection: %s < f < %s Hz'%(str(fcalib_lo), str(fcalib_hi)))
print('.. whitening FFT (window, stride):',whiten_fft, 's')
#print('.. bandpass selection: %d < f < %d Hz'%(fsg_lo, fsg_hi))
print('.. bandpass selection: %s < f < %s Hz'%(str(fsg_lo), str(fsg_hi)))
print('.. inject signal? %s'%str(inj_sg))
print('.. add spin? %s'%str(spin))

###################################################################
# Read in inputs

print('>> Input files...')
inputfiles = {}
for k in ks:
    print('.. detector:',k)
    inputfiles[k] = open(inputlist[k]).readlines()#[1:3]
    inputfiles[k] = [f.strip() for f in inputfiles[k]]
    for f in inputfiles[k]:
        print('..',f)
        assert os.path.exists(f)

print('>> Processing GPS times...')
start = int(inputfiles[kH1][0].split('/')[-1].split('-')[-2])
end = int(inputfiles[kH1][-1].split('/')[-1].split('-')[-2])
print('.. start: %d s'%start)
print('.. end: %d + %d s'%(end, end_offset))
end += end_offset
print('.. end: %d s'%end)

#seed_from_file = int(inputfiles[kH1][0].split('-')[-2])
seed_from_file = start
print('.. rand seed to be used:',seed_from_file)

###################################################################
# Check input file channels
kdq, kinj, kdata = {}, {}, {}

for k in ks:
    kdq[k], kinj[k], kdata[k] = get_channel_names(inputfiles[k][0])
    print(kdq[k], kinj[k], kdata[k])
    print('>> Available channels:')
    print('.. strain data :',kdata[k])
    print('.. data quality:',kdq[k])
    print('.. hardware inj:',kinj[k])

###################################################################
# Read in gwf frames
series, series_dq, series_inj = {}, {}, {}

print('>> Reading frames...')
for k in ks:
    series[k] = TimeSeries.read(inputfiles[k], kdata[k], start=start, end=end)
    series_dq[k] = TimeSeries.read(inputfiles[k], kdq[k], start=start, end=end)
    series_inj[k] = TimeSeries.read(inputfiles[k], kinj[k], start=start, end=end)

    print(series[k])
    print('.. sample_rate:',series[k].sample_rate)

    # Timestamps should be same for H1 and L1 file, so only need to store
    # value from one of them
    if k != kH1: continue

    # see: https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries.html#gwpy.timeseries.TimeSeries
    t0 = series[k].t0.value
    dt = series[k].dt.value
    t_total = series[k].duration.value
    tf = t0 + t_total
    print('.. [t0, tf): [%f, %f) s'%(t0, tf))
    print('.. duration: %f s'%t_total)
    print('.. dt: %f s'%dt)
    print('.. len(series):',len(series[k]))
    #assert t_total == end-start
    assert len(series[k]) == series[k].duration.value/series[k].dt.value

    times = series[k].times.value
    print('.. GPS times:',series[k].times)
    print('.. time span:',series[k].xspan)
    print('.. times[ 0]:',series[k].times[0])
    print('.. times[-1]:',series[k].times[-1])
    print('.. len(times):',len(times))
    assert len(series[k]) == len(times)

# Verify timestamps are indeed identical
assert t0 == series[kL1].t0.value
assert dt == series[kL1].dt.value
assert t_total == series[kL1].duration.value
assert np.array_equal(times, series[kL1].times.value)

# Calculate effective time window
t_trunc   = 2.5 #2. # s: how much to truncate at start and end of series after whitening, to remove artifacts
n_trunc   = int(t_trunc//dt) # N elements in t_trunc
t_eff     = t_window - 2*(t_trunc)
print('.. amount to truncate on either side of waveform, after whitening: %2.1f s (%d elements)'%(t_trunc, n_trunc))
print('.. effective waveform window per time segment: %d s (%d elements)'%(t_eff, t_eff//dt))

###################################################################
# Initialize pq output params
print('>> Initializing output...')
# Make output dir for plots
outlist = inputlist[kH1].split('/')[-1].split('.')[0].replace('_list','') # cleaned name of input list
outlist = outlist.replace(kH1, '-'.join(ks))
outdir  = 'plots/%s'%(outlist)
if not os.path.isdir(outdir):
    os.makedirs(outdir)
# Make filename str
outname = inputfiles[kH1][0].split('/')[-1].split('.')[0].split('-')[1] # cleaned input file dataset family
outname = '%s_%s'%('-'.join(ks), '_'.join(outname.split('_')[1:])) # prepend list of detectors used
outplots = '%s/%s'%(outdir, outname)
print('.. will save plots to:', outplots)

# Make output dir for parquet
# BBH mass and distance ranges
mlo, mhi = 10., 50.#10., 50. # recommended total mass > 20 for generators: https://arxiv.org/pdf/2003.12079.pdf
dlo, dhi = 100., 500.#500. #5.e2, 1.e3
#expt = 'm20to50_d1E2to5E2_teff%2.1fs_tw%2.1fs_ts%2.1fs_nobp'%(t_eff, t_window, t_stride)
#expt = 'm20to50_d1E2to5E2_teff%2.1fs_tw%2.1fs_ts%2.1fs'%(t_eff, t_window, t_stride)
expt = 'm%dto%d_d%.Eto%.E_teff%2.1fs_tw%2.1fs_ts%2.1fs'%(mlo, mhi, dlo, dhi, t_eff, t_window, t_stride)
expt = expt.replace('+0','')
if spin:
    expt += '_spin'
if nowht:
    expt += '_nowht'
if nobp:
    expt += '_nobp'
outdir = 'data/%s/parquet/%s'%(inputfiles[kH1][0].split('/')[-3],expt) # observation run
if not os.path.isdir(outdir):
    os.makedirs(outdir)
# Make output filename for parquet
pqout = '%s/%s_%s.parquet'%(outdir, outlist, 'sgbg' if inj_sg else 'bg')
print('.. parquet output:',pqout)

###################################################################
# Set segment params
# For this job, set random seed based on first input file GPS time
np.random.seed(seed_from_file)
#from numpy.random import MT19937
#from numpy.random import RandomState, SeedSequence
#rs = RandomState(MT19937(SeedSequence(seed_from_file)))
print(np.random.randint(10, size=5))
print(np.random.randint(10, size=5))

# Choose a GPS end time, sky location, and polarization phase for the merger
# NOTE: Right ascension and polarization phase runs from 0 to 2pi
#       Declination runs from pi/2. to -pi/2 with the poles at pi/2. and -pi/2.
# See known events: https://www.gw-openscience.org/eventapi/html/allevents/
#
print('>> Making time segments via striding...')

window = int(t_window//dt) # steps
stride = int(t_stride//dt) # steps
assert t_window <= t_total
assert t_window >= t_stride
print('.. len(window):', window)
print('.. len(stride):', stride)

# Get stride params
byte_strides = (times.strides[0]*stride, times.strides[0])
nsegs = (len(times)-window)/stride + 1
print('.. N segments:', nsegs)
if nsegs%1 != 0:
    print(nsegs)
    print('!! WARNING: series will be truncated!')
nsegs = int(np.floor(nsegs))
print('.. N segments:', nsegs)

# Do actual strides to make time segments
timesegs = as_strided(times, shape=(nsegs, window), strides=byte_strides)
print('.. timesegs[ 0][ 0]: %f s'%timesegs[0][0])
print('.. timesegs[-1][-1]: %f s'%timesegs[-1][-1])
assert nsegs == len(timesegs)

###################################################################
# Set BBH sim params
kdefault = -99
if inj_sg:
    # Make uniform distn
    # [a, b) ~ (b-a)*np.random.random(size) + a

    # Time placement of GW signal
    # defines *end time* of GW signal
    t_pad_lo   = 0.2  #0.25  # s how near to left end of t_eff to place signal
    t_pad_hi   = 0.05 #0.1   # s how near to right end of t_eff to place signal
    t_gwoff_lo = t_trunc+t_eff-t_pad_lo+dt
    t_gwoff_hi = t_trunc+t_pad_hi+dt
    #print(t_gwoff_lo, t_gwoff_hi)
    # offset to apply from end time of noise waveform
    t_gwoffs   = (t_gwoff_hi-t_gwoff_lo)*np.random.random(nsegs) + t_gwoff_lo #-24 # end of GW
    #print(t_gwoff_lo, t_gwoff_hi, t_gwoffs[-5:])

    # BBH phase space
    '''
    # Defined above before parquet output str
    mlo, mhi = 10., 50.#10., 50.
    dlo, dhi = 100., 500.#500. #5.e2, 1.e3
    '''
    m1s   = (mhi-mlo)*np.random.random(nsegs) + mlo    # in M_sun. GW150914: ~35M_sun. [10, 100] -> [10, 50]
    m2s   = (mhi-mlo)*np.random.random(nsegs) + mlo    # in M_sun. GW150914: ~30M_sun. [10, 100] -> [10, 50]
    sp1s  = 0.99*np.random.random(nsegs) if spin else np.zeros(nsegs)   # spinz1 (0., 0.99)
    sp2s  = 0.99*np.random.random(nsegs) if spin else np.zeros(nsegs)   # spinz2 (0., 0.99)
    incs  = np.pi*np.random.random(nsegs) if spin else np.zeros(nsegs)  # 0. # inclination of BBH orbital plane: [0, pi]
    coas  = 2.*np.pi*np.random.random(nsegs) if spin else np.zeros(nsegs)  # 0. # coalescence phase: [0, 2pi]
    dists = (dhi-dlo)*np.random.random(nsegs) + dlo    # 340Mpc, distance to detector, in Mpc, [1e2, 5e3], Andromeda galaxy about ~< 1 Mpc away from Earth.
    #decs  = 2*np.pi*np.random.random(nsegs) - np.pi/2. #0.65 # declination: [-pi/2, pi/2)
    #ras   = 2*np.pi*np.random.random(nsegs)            #0. # right ascension: [0, 2pi]
    #pols  = 2*np.pi*np.random.random(nsegs) - np.pi/2. #0. # polarization: [-pi, pi) or [0, 2pi] (controls phase of wave)
    decs  = np.pi*np.random.random(nsegs) - np.pi/2. #0.65 # declination: [-pi/2, pi/2)
    ras   = 2.*np.pi*np.random.random(nsegs)            #0. # right ascension: [0, 2pi]
    pols  = 2.*np.pi*np.random.random(nsegs) - np.pi #0. # polarization: [-pi, pi) or [0, 2pi] (controls phase of wave)

###################################################################
# Make time segments
print('>> Looping over time segments...')

# dicts for holding batches of each detector
batch, dqs, injs = {}, {}, {}
sg, sb, wav = {}, {}, {}
# dict for holding output values to parquet
data = {}

nprint = 1000  # Frequency of debug prints
nwrite = 0   # N segments actually written out after quality cuts

now = time.time()
for i in range(nsegs):

    if i%nprint == 0: print('.. %d / %d'%(i, nsegs))
    if i>20:
        pass
        #break

    # Get time window for this segment
    tseg = timesegs[i]
    tstart, tend = tseg[0], tseg[-1]+dt # stride generates [t0, t1), so need to add +dt to get full t_window
    assert t_window == tend-tstart

    # Get data for this segment
    skip_batch = False
    for k in ks:
        batch[k] = series[k].crop(tstart, tend)
        dqs[k]   = series_dq[k].crop(tstart, tend)
        injs[k]  = series_inj[k].crop(tstart, tend)
        assert batch[k].duration.value == t_window
        #print(batch[0], batch[1], batch[2], batch[3])

        # Quality cuts
        # NOTE: only needs to be checked for effective window
        # i.e. full window less truncated segments
        # Make sure no NaNs
        if np.any(np.isnan(batch[k].value[n_trunc:-n_trunc])):
            skip_batch = True
        # Make sure DQ:CAT3 for both CBC and burst
        if np.any(dqs[k].value[n_trunc:-n_trunc] != 127):
            skip_batch = True
        # Make sure there were no hardware injs
        if np.any(injs[k].value[n_trunc:-n_trunc] != 31):
            skip_batch = True

    if skip_batch: continue

    assert np.array_equal(batch[kH1].times.value, batch[kL1].times.value)

    debug = True if i%nprint == 0 else False
    debug = False

    # Waveform processing
    if inj_sg:
        for k in ks:
            det = detstr(k)
            # Generate GW signal
            saveplot = '%s_%s_sg_seg%05d.pdf'%(outplots, det, i)
            #sg[k] = make_bbh(m1s[i], m2s[i], dists[i], decs[i], ras[i], pols[i], tend-t_gwoffs[i], fcalib_lo, dt, debug=debug, saveplot=saveplot, kdet=k)
            sg[k] = make_bbh_spin(m1s[i], m2s[i], sp1s[i], sp2s[i], incs[i], coas[i], dists[i], decs[i], ras[i], pols[i], tend-t_gwoffs[i], fcalib_lo, dt, debug=debug, saveplot=saveplot, kdet=k)
            # Inject signal into data
            sb[k] = batch[k].inject(sg[k])
            # Process waveform
            saveplot = '%s_%s_sgbg_seg%05d.pdf'%(outplots, det, i)
            wav[k] = process_strain(sb[k], w=whiten_fft, fcalib_lo=fcalib_lo, fsg_lo=fsg_lo, fsg_hi=fsg_hi, t_trunc=t_trunc, dt=dt, debug=debug, saveplot=saveplot, det=det)
    else:
        for k in ks:
            det = detstr(k)
            # Process waveform
            saveplot = '%s_%s_bg_seg%05d.pdf'%(outplots, det, i)
            wav[k] = process_strain(batch[k], w=whiten_fft, fcalib_lo=fcalib_lo, fsg_lo=fsg_lo, fsg_hi=fsg_hi, t_trunc=t_trunc, dt=dt, debug=debug, saveplot=saveplot, det=det)

    # Check for NaNs again
    skip_batch = False
    for k in ks:
        if np.any(np.isnan(wav[k].value)):
            skip_batch = True
    if skip_batch: continue

    # Write to parquet
    # First, convert to dict
    data['strain'] = np.stack([wav[kH1].value, wav[kL1].value], axis=0)
    data['t0']     = wav[kH1].times.value[0]
    if inj_sg:
        data['y']    = 1.
        data['m1']   = m1s[i]
        data['m2']   = m2s[i]
        data['sp1']  = sp1s[i]
        data['sp2']  = sp2s[i]
        data['inc']  = incs[i]
        data['coa']  = coas[i]
        data['dist'] = dists[i]
        data['dec']  = decs[i]
        data['ra']   = ras[i]
        data['pol']  = pols[i]
        data['toff'] = t_eff-t_gwoffs[i] # store offset from start time of this waveform instead
    else:
        data['y']    = 0.
        data['m1']   = kdefault
        data['m2']   = kdefault
        data['sp1']  = kdefault
        data['sp2']  = kdefault
        data['inc']  = kdefault
        data['coa']  = kdefault
        data['dist'] = kdefault
        data['dec']  = kdefault
        data['ra']   = kdefault
        data['pol']  = kdefault
        data['toff'] = kdefault
    # Then convert to pq arrays
    pqdata = [pa_array(d) for d in data.values()]
    # Then convert to pq table
    table = pa.Table.from_arrays(pqdata, list(data.keys()))
    # Write table to pq file
    if nwrite == 0:
        writer = pq.ParquetWriter(pqout, table.schema, compression='snappy')
    writer.write_table(table)

    nwrite += 1

if nwrite > 0:
    writer.close()
proc_time = time.time() - now
proc_time_str = str('%f s'%proc_time if proc_time <= 60. else '%f min'%(proc_time/60.))
print('>> Done.')
#print('deltat:',deltats[1]-deltats[0])
print('.. nwrite / nsegs: %d / %d'%(nwrite, nsegs))
#print('.. processing time: %s'%('%f s'%proc_time if proc_time <= 60 else '%f min'%proc_time/60.))
print('.. processing time: %s'%proc_time_str)

# Do a read test
if nwrite > 0:
    pqin = pq.ParquetFile(pqout)
    print(pqin.metadata)
    print(pqin.schema)
    X = pqin.read_row_group(0, columns=['t0','y','m1','m2']).to_pydict()
    print(X)
    Xstrain = pqin.read_row_group(0, columns=['strain']).to_pydict()
    print(np.float32(Xstrain['strain']).shape)
