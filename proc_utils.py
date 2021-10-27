from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
#import pycbc as pycbc
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector

import pyarrow as pa
import numpy as np

def detstr(k):
    if k == 'H1':
        return 'Hanford'
    elif k == 'L1':
        return 'Livingston'
    else:
        raise Exception('Unknown det:%s'%k)

def plot_series(series, xlims=None, ylims=None, title='time series', ylabel='Amplitude [strain]', saveplot=None, det='Hanford'):
    plot = series.plot(title='LIGO-%s, %s'%(det, title), ylabel=ylabel, color='gwpy:ligo-%s'%det.lower())
    #print(plot.axes[0])
    if xlims is not None:
        plot.axes[0].set_xlim(xlims[0], xlims[1])
    if ylims is not None:
        plot.axes[0].set_ylim(ylims[0], ylims[1])
    #plot.show()
    if saveplot is not None:
        plot.save(saveplot)
    #    plot.close()

def plot_asd(series, w=(4,2), xlims=None, ylims=None, det='Hanford'):
    asd = series.asd(*w)
    # => delta_f = 1/4 = 0.25 Hz
    plot = asd.plot(title='LIGO-%s, amplitude spectral density (ASD)'%det, ylabel=r'ASD [$\sqrt{Hz}$]', color='gwpy:ligo-%s'%det.lower())
    #print(plot.axes[0])
    if xlims is not None:
        plot.axes[0].set_xlim(xlims[0], xlims[1])
    if ylims is not None:
        plot.axes[0].set_ylim(ylims[0], ylims[1])
    plot.show()

def plot_psd(series, w=(4,2), xlims=None, ylims=None, det='Hanford'):
    # see: https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries.html#gwpy.timeseries.TimeSeries.psd
    psd = series.psd(*w) # = (total_time_duration=sgdelta_t, stride_overlap=2 (default calculated internally as appropriate for window fn))
    # => delta_f = 1/4 = 0.25 Hz
    plot = psd.plot(title='LIGO-%s, power spectral density (PSD)'%det, ylabel=r'PSD [Hz]', color='gwpy:ligo-%s'%det.lower())
    #plot.axes[1].set_ylim(1e-20, 1e-15)
    if xlims is not None:
        plot.axes[0].set_xlim(xlims[0], xlims[1])
    if ylims is not None:
        plot.axes[0].set_ylim(ylims[0], ylims[1])
    plot.show()
    return psd

def _fft_length_default(dt):
    # from https://github.com/gwpy/gwpy/blob/26f63684db17104c5d552c30cdf01248b2ec76c9/gwpy/timeseries/timeseries.py#L44
    """Choose an appropriate FFT length (in seconds) based on a sample rate
    Parameters
    ----------
    dt : `~astropy.units.Quantity`
        the sampling time interval, in seconds
    Returns
    -------
    fftlength : `int`
        a choice of FFT length, in seconds
    """
    return int(max(2, np.ceil(2048 * dt)))

def whiten_series(series, w=(4,2), t_trunc=None, dt=1./4096., debug=False):
    # pycbc
    # pycbc does truncation by default
    # comparing: https://github.com/gwpy/gwpy/blob/v2.0.4/gwpy/timeseries/timeseries.py#L1669
    # and https://pycbc.org/pycbc/latest/html/gw150914.html?highlight=whiten + https://pycbc.org/pycbc/latest/html/_modules/pycbc/types/timeseries.html#TimeSeries.whiten
    #series_pycbc = series.to_pycbc()
    #wht = series_pycbc.whiten(series.duration.value, _fft_length_default(series.dt.value))
    #wht = TimeSeries.from_pycbc(wht)
    # gwpy
    wht = series.whiten(*w)
    # truncate corrupted ends
    # by `t_trunc` seconds on each end
    # from: https://pycbc.org/pycbc/latest/html/_modules/pycbc/types/timeseries.html#TimeSeries.whiten
    if t_trunc is None or t_trunc == 0.:
        return wht
    elif t_trunc == -1:
        max_filter_duration = _fft_length_default(series.dt.decompose().value)
        max_filter_len = int(max_filter_duration * series.sample_rate.decompose().value)
        t_len = int(max_filter_len/2)
        wht = wht[t_len:-t_len]
        return wht
    else:
        t_len = int(t_trunc/dt)
        if debug: print('   .. whitened series will be truncated on either side by %f s (%d elements)'%(t_trunc, t_len))
        wht = wht[t_len:-t_len]
        return wht

def filter_band_notch(wht, fsg_lo=35., fsg_hi=250.):
    # define bandpass
    bp = filter_design.bandpass(fsg_lo, fsg_hi, wht.sample_rate)
    # define notches
    notches = [filter_design.notch(line, wht.sample_rate) for line in (60, 120, 180)]
    # chain filters
    zpk = filter_design.concatenate_zpks(bp, *notches)
    # set filtfilt=True to filter both backwards and forwards to preserve the correct phase at all frequencies
    # see also: https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries.html#gwpy.timeseries.TimeSeries.filter
    bp = wht.filter(zpk, filtfilt=True)
    #'''
    return bp

def process_strain(series, w=(4,2), fcalib_lo=16., fcalib_hi=2048., fsg_lo=35., fsg_hi=250., t_trunc=None, dt=1./4096., debug=False, saveplot=None, det='Hanford'):

    # Apply calib highpass
    # NOTE: O1 4KHz data already has a 2 kHz cutoff
    #series_bp = series.bandpass(fcalib_lo, fcalib_hi)
    series_bp = series.highpass(fcalib_lo) #.lowpass(fcalib_hi)
    if debug:
        pass
        #plot_series(series_bp, ylims=[-7.5e-19, 7.5e-19], det=det)

    # Whiten
    if w is None:
        return series_bp
    else:
        wht = whiten_series(series_bp, w=w, t_trunc=t_trunc, dt=dt)
        if debug:
            pass
            #plot_series(wht)#, ylims=[-10., 10.], det=det)
            #plot_asd(wht, w=(4,2), xlims=[fcalib_lo, fcalib_hi], det=det)#, ylims=[1.e-28, 1.e-19])
            print('   .. whitened series duration:', wht.duration)

    # Apply analysis bandpass
    if fsg_lo is None or fsg_hi is None:
        return wht
    else:
        bp = filter_band_notch(wht, fsg_lo, fsg_hi)
        if debug:
            pass
            plot_series(bp, ylims=[-5., 5.], saveplot=saveplot, det=det)
        return bp


def make_bbh_spin(m1, m2, sp1, sp2, inc, coa, dist, dec, ra, pol, t_end, fcalib_lo=16., dt=1./4096., debug=False, saveplot=None, kdet='H1'):

    # Make waveform
    # from: https://pycbc.org/pycbc/latest/html/detector.html#antenna-patterns-and-projecting-a-signal-into-the-detector-frame
    # and https://pycbc.org/pycbc/latest/html/waveform.html#generating-one-waveform-in-multiple-detectors
    # API: https://pycbc.org/pycbc/latest/html/pycbc.waveform.html#pycbc.waveform.waveform.get_td_waveform

    if debug:
        print('   .. m1:%f, m2:%f, sp1:%f, sp2:%f, inc:%f, coa:%f, dist:%f, dec:%f, ra:%f, pol:%f, t_end:%f'%(m1, m2, sp1, sp2, inc, coa, dist, dec, ra, pol, t_end))

    assert kdet == 'H1' or kdet == 'L1'
    this_det = Detector(kdet)
    #this_det = Detector(det[0]+'1')
    #det_h1 = Detector('L1')
    #det_l1 = Detector('L1')
    #det_v1 = Detector('V1')

    # We get back the fp and fc antenna pattern weights.
    # These factors allow us to project a signal into what the detector would
    # observe
    fp, fc = this_det.antenna_pattern(ra, dec, pol, t_end)

    # Generate the waveform
    hp, hc = get_td_waveform(approximant='SEOBNRv4',
                             mass1=m1, # in M_sun. GW150914: ~35M_sun
                             mass2=m2, # in M_sun. GW150914: ~35M_sun
                             spin1z=sp1,
                             spin2z=sp2,
                             inclination=inc,#1.23, # [0 (default), pi) # angle between the orbital angular momentum L and the line-of-sight at the reference frequency
                             coa_phase=coa, # https://pycbc.org/pycbc/latest/html/pycbc.waveform.html#pycbc.waveform.waveform.get_td_waveform [0, 2pi)
                             delta_t=dt,
                             distance=dist, # distance from detector, in Mpc: GW150914: ~300Mpc
                             f_lower=fcalib_lo)
                             #f_lower=40)

    ## Apply the factors to get the detector frame strain
    ht = fp*hp + fc*hc

    # The projection process can also take into account the rotation of the
    # earth using the project wave function.
    hp.start_time += t_end
    hc.start_time += t_end
    ht = this_det.project_wave(hp, hc,  ra, dec, pol)
    #signal_h1 = det_h1.project_wave(hp, hc,  right_ascension, declination, polarization)
    #signal_l1 = det_l1.project_wave(hp, hc,  right_ascension, declination, polarization)
    #signal_v1 = det_v1.project_wave(hp, hc,  right_ascension, declination, polarization)

    #ht.resize(len(series))
    gw = TimeSeries.from_pycbc(ht)
    if debug:
        plot_series(gw, saveplot=saveplot, det=detstr(kdet))
        print('   .. GW, max strain:',gw.max())
    return gw

def make_bbh(m1, m2, dist, dec, ra, pol, t_end, fcalib_lo=16., dt=1./4096., debug=False, saveplot=None, kdet='H1'):

    # Make waveform
    # from: https://pycbc.org/pycbc/latest/html/detector.html#antenna-patterns-and-projecting-a-signal-into-the-detector-frame
    # and https://pycbc.org/pycbc/latest/html/waveform.html#generating-one-waveform-in-multiple-detectors

    if debug:
        print('   .. m1:%f, m2:%f, dist:%f, dec:%f, ra:%f, pol:%f, t_end:%f'%(m1, m2, dist, dec, ra, pol, t_end))

    assert kdet == 'H1' or kdet == 'L1'
    this_det = Detector(kdet)
    #this_det = Detector(det[0]+'1')
    #det_h1 = Detector('L1')
    #det_l1 = Detector('L1')
    #det_v1 = Detector('V1')

    # We get back the fp and fc antenna pattern weights.
    # These factors allow us to project a signal into what the detector would
    # observe
    fp, fc = this_det.antenna_pattern(ra, dec, pol, t_end)

    # Generate the waveform
    hp, hc = get_td_waveform(approximant='SEOBNRv4',
                             mass1=m1, # in M_sun. GW150914: ~35M_sun
                             mass2=m2, # in M_sun. GW150914: ~35M_sun
                             #spin1z=0.9,
                             #spin2z=0.4,
                             inclination=0.,#1.23, # [0 (default), pi) # angle between the orbital angular momentum L and the line-of-sight at the reference frequency
                             #coa_phase=2.45, # [0, 2pi)
                             delta_t=dt,
                             distance=dist, # distance from detector, in Mpc: GW150914: ~300Mpc
                             f_lower=fcalib_lo)
                             #f_lower=40)

    ## Apply the factors to get the detector frame strain
    ht = fp*hp + fc*hc

    # The projection process can also take into account the rotation of the
    # earth using the project wave function.
    hp.start_time += t_end
    hc.start_time += t_end
    ht = this_det.project_wave(hp, hc,  ra, dec, pol)
    #signal_h1 = det_h1.project_wave(hp, hc,  right_ascension, declination, polarization)
    #signal_l1 = det_l1.project_wave(hp, hc,  right_ascension, declination, polarization)
    #signal_v1 = det_v1.project_wave(hp, hc,  right_ascension, declination, polarization)

    #ht.resize(len(series))
    gw = TimeSeries.from_pycbc(ht)
    if debug:
        plot_series(gw, saveplot=saveplot, det=detstr(kdet))
        print('   .. GW, max strain:',gw.max())
    return gw

def plot_qt(qt, xlims=None, ylims=None, det='Hanford'):
    plot = qt.plot(title='LIGO-%s, Q-transform spectrogram'%det, yscale='log')
    #print(plot.axes[0])
    if xlims is not None:
        plot.axes[0].set_xlim(xlims[0], xlims[1])
    if ylims is not None:
        plot.axes[0].set_ylim(ylims[0], ylims[1])
    plot.colorbar(cmap='viridis', label='Normalized energy')
    plot.show()

def pa_array(d):
    arr = pa.array([d]) if np.isscalar(d) or type(d) == list else pa.array([d.tolist()])
    #print(arr.type)
    ## double to single float
    if arr.type == pa.float64():
        arr = arr.cast(pa.float32())
    elif arr.type == pa.list_(pa.float64()):
        arr = arr.cast(pa.list_(pa.float32()))
    elif arr.type == pa.list_(pa.list_(pa.float64())):
        arr = arr.cast(pa.list_(pa.list_(pa.float32())))
    elif arr.type == pa.list_(pa.list_(pa.list_(pa.float64()))):
        arr = arr.cast(pa.list_(pa.list_(pa.list_(pa.float32()))))
    elif arr.type == pa.list_(pa.list_(pa.list_(pa.list_(pa.float64())))):
        arr = arr.cast(pa.list_(pa.list_(pa.list_(pa.list_(pa.float32())))))
    #else:
    #    print('Unknown type for conversion to (list of) floats',arr.type)
    #print(arr.type)
    return arr
