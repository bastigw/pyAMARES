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
import numpy as np
import pyAMARES

pyAMARES.__version__
```

```{code-cell} ipython3
priorknowledge = pyAMARES.initialize_FID(
    fid=None, priorknowledgefile="example_human_brain_31P_7T.csv", preview=True
)
```

```{code-cell} ipython3
from copy import deepcopy

params0 = deepcopy(
    priorknowledge.initialParams
)  # Make a copy of initialParams to be perturbed
```

```{code-cell} ipython3
def perturb_value(value, percentage=5):
    percentage = float(percentage)
    # Generate a random perturbation factor between 0.95 and 1.05
    factor = np.random.uniform(1 - percentage / 100, 1 + percentage / 100)
    # Apply the perturbation factor
    result = value * factor
    # print(f"Perturbing input {value=} to {result=}")
    return result


def perturb_table(inputparams, percentage=5, freq_shift=5, phase_shift=0):
    params = deepcopy(inputparams)
    for i in params:
        if params[i].name.startswith("ak") or params[i].name.startswith("dk"):
            params[i].value = perturb_value(params[i].value)
        if params[i].name.startswith("freq"):
            params[i].value += np.random.uniform(-freq_shift, freq_shift)
        if params[i].name.startswith("phi"):
            params[i].value += np.random.uniform(
                -np.deg2rad(phase_shift), np.deg2rad(phase_shift)
            )
    return params
```

```{code-cell} ipython3
paramlist = []
for i in range(8):
    params = perturb_table(params0, percentage=5, freq_shift=5, phase_shift=0)
    paramlist.append(params)
```

```{code-cell} ipython3
fidlist = []
for params in paramlist:
    fid = pyAMARES.kernel.fid.simulate_fid(
        params,
        MHz=120.0,
        sw=10000.0,
        deadtime=200e-6,
        fid_len=1024,
        snr_target=20,
        preview=False,
    )
    fidlist.append(fid)
```

```{code-cell} ipython3
FIDobj = pyAMARES.initialize_FID(
    fid=fidlist[0],
    MHz=120.0,
    sw=10000.0,
    deadtime=200e-6,
    normalize_fid=False,
    priorknowledgefile="example_human_brain_31P_7T.csv",
    preview=False,
)
```

```{code-cell} ipython3
out1 = pyAMARES.fitAMARES(
    fid_parameters=FIDobj,
    fitting_parameters=FIDobj.initialParams,
    method="leastsq",
    ifplot=True,
)
```

```{code-cell} ipython3
out1.styled_df
```

```{code-cell} ipython3
fidarr = np.array(fidlist)
fidarr.shape
```

```{code-cell} ipython3
result_list = pyAMARES.run_parallel_fitting_with_progress(
    fidarr,  # 2D array of FIDs. Here, `fid3.shape=(366,1024)` indicates 366 FIDs, each with 1024 points.
    FIDobj_shared=out1,  # Use the FID object `out2` for fitting all FIDs.
    initial_params=out1.fittedParams,  # Use the fitted results of the first FID as the initial parameters.
    num_workers=4,  # Parallel processing with 2 sessions if used in Google Colab, suitable for the 2 CPUs available in Google Colab.
    initialize_with_lm=True,
    method="leastsq",
)  # Use the Levenberg-Marquardt method by default for faster processing.
```

```{code-cell} ipython3
pyAMARES.highlight_dataframe(result_list[0])
```

```{code-cell} ipython3

```
