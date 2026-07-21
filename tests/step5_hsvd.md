---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell} ipython3
import pyAMARES
```

# Fitting MRS with unknown species, using HSVDinitializer
**[Try this tutorial on Google Colab!](https://colab.research.google.com/drive/15zKm0rqnVheYwk-5D2orwNalTDu5a6zg)**

- First, simulate a spectrum with two peaks

```{code-cell} ipython3
priorknowledge = pyAMARES.initialize_FID(
    fid=None, priorknowledgefile="singlet.csv", preview=True
)
```

- **Simulate an MRS Spectra Using Scanner Parameters**:
    - **MHz (Field Strength)**: 300 MHz. 
    - **sw (Spectral Width)**: 5000.0 Hz.
    - **Deadtime**: 100 microseconds (100e-6 seconds).
    - **Number of Points (fid_len)**: 1024
    - **SNR (Signal to Noise Ratio, snr_target)**: 40.

```{code-cell} ipython3
fid = pyAMARES.kernel.fid.simulate_fid(
    priorknowledge.initialParams,
    MHz=300.0,
    sw=5000.0,
    deadtime=100e-6,
    fid_len=1024,
    snr_target=40,
    preview=True,
)
```

```{code-cell} ipython3
FIDobj = pyAMARES.initialize_FID(
    fid=fid,
    MHz=300.0,
    sw=5000.0,
    deadtime=100e-6,
    priorknowledgefile=None,
    preview=True,
    normalize_fid=False,
)
```

```{code-cell} ipython3
params_hsvd = pyAMARES.HSVDinitializer(
    fid_parameters=FIDobj, num_of_component=2, fitting_parameters=None, preview=True
)
```

```{code-cell} ipython3
params_hsvd
```

```{code-cell} ipython3
FIDresult = pyAMARES.fitAMARES(
    fid_parameters=FIDobj,
    fitting_parameters=params_hsvd,
    method="least_squares",
    ifplot=True,
)
```

```{code-cell} ipython3
FIDresult.styled_df
```

```{code-cell} ipython3
FIDobj.ppm
```

# Frequency-Selective AMARES

+++

## Method 1: frequency-selective AMARES using MPFIR filter 
   - References: 
       1. Vanhamme et al, J Mag Reson 143, 1-16(2000)
       2. Sundin et al, J Mag Reson 139, 189-204 (1999)

```{code-cell} ipython3
from pyAMARES import filter_param_by_ppm
```

```{code-cell} ipython3
fit_ppm = (-0.5, 8.2)  # ppm
```

```{code-cell} ipython3
FIDobj2 = pyAMARES.filter_fid_by_ppm(FIDobj, fit_ppm=fit_ppm, ifplot=True)
```

```{code-cell} ipython3
param2 = filter_param_by_ppm(params_hsvd, fit_ppm=fit_ppm, MHz=FIDobj2.MHz)
```

```{code-cell} ipython3
FID_result_positive_ppm_peak = pyAMARES.fitAMARES(
    fid_parameters=FIDobj2,  # Filtered FID object
    fitting_parameters=param2,
    method="least_squares",
    ifplot=True,
)
```

```{code-cell} ipython3
FID_result_positive_ppm_peak.styled_df
```

## Method 2: Frequency-Selective AMARES using Objective Function with frequency range

```{code-cell} ipython3
FID_result_positive_ppm_peak2 = pyAMARES.fitAMARES(
    fid_parameters=FIDobj,
    fitting_parameters=param2,
    fit_range=fit_ppm,  # Instead of filtering out the spectrum,  (-.5, 8.2) ppm
    # is passed to the objective_func
    objective_func=pyAMARES.objective_range,  # This objective_range can accept the `fit_range` argument
    method="least_squares",
    ifplot=True,
)
```

```{code-cell} ipython3
FID_result_positive_ppm_peak2.styled_df
```

```{code-cell} ipython3

```
