# gw-analysis
Tools for the analysis of LIGO gravitational wave data from the Gravitational Wave Open Science Center (GWOSC) [1].
#### Machine Learning (ML) workflow:
The workflow below describes the process for trainining a ResNet-based convolutional neural network (CNN) to distinguish GWOSC data containing no known gravitational wave (GW) signals ("background" class), from those containing a simulated binary blackhole merger (BBH) ("signal" class).
###### 1. Time series to image data conversion
The following script is used to first convert the time series data in the GWOSC `gwf` format files to discrete waveform images suitable for training and inference,
`convert_gwf2pq.py`.
The output is saved in a `pyarrow.parquet` format file [2]. The above script contains options for converting the time series data as is, as well as for overlaying simulated BBH signals on top of them. Supplementary processing functions are found in `proc_utils.py`. These are based off the `PyCBC` [3] and `GWpy` [4] packages.
###### 2. ML training
Once the GWOSC data have been converted to `parquet` format, the training script
`bbh_trainer.py`
reads these and performs the actual ML model training/optimization. A number of options are included for modifying the ML model and data preprocessing hyperparameters. Supplementary functions for data feeding (`data_utils.py`), ML model definition (`network_utils.py`), and basic validation/inference (`eval_utils.py`) are also provided. The above scripts are implemented using the `PyTorch` deep learning package [5].
###### 2. Analysis
Various notebooks are included for performing more in-depth analyses of the BBH predictions from the trained ML model. These files are prefixed with `bbh_eval*`. For instance, for making basic distributions of the BBH predictions converted to a detection statistic, see `bbh_eval-nnout.ipynb`. These notebooks typically re-use some of the earlier supplementary functions from the training step.

#### References:
[1] Gravitational Wave Open Science Center. https://www.gw-openscience.org/.
[2] Apache PyArrow Parquet format. https://arrow.apache.org/docs/python/parquet.html.
[3] PyCBC. http://pycbc.org/pycbc/latest/html/.
[4] GWpy. https://gwpy.github.io/docs/latest/.
[5] PyTorch. https://pytorch.org/.
