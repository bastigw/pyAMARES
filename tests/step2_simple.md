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
import numpy as np

pyAMARES.__version__
```

# Fitting Simulated In Vivo 31P MRS Data
**[Try This Tutorial on Google Colab!](https://colab.research.google.com/drive/1H8GdP4XX292JovAF5TNOEBaYQT5vk_DT)**

+++

## Simulating an In Vivo MRS Spectrum

+++

- **Load Prior Knowledge**: Use the dataset based on the 7T brain data reported by Ren et al. in NMR Biomedicine, 28(11): 1455–1462.

```{code-cell} ipython3
priorknowledge = pyAMARES.initialize_FID(
    fid=None, priorknowledgefile="example_human_brain_31P_7T.csv", preview=True
)
```

- **Perturb Peak Parameters**: Randomly adjust the 31P spectra peak parameters by 5%, chemical shift by 10 Hz

```{code-cell} ipython3
from copy import deepcopy
```

```{code-cell} ipython3
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
params = perturb_table(params0, percentage=5, freq_shift=5, phase_shift=0)
```

- **Simulate the 31P MRS Spectra Using Scanner Parameters**:
    - **MHz (Field Strength)**: 120 MHz, corresponding to 31P at 7T.
    - **sw (Spectral Width)**: 10000.0 Hz.
    - **Deadtime**: 200 microseconds (200e-6 seconds).
    - **Number of Points (fid_len)**: 1024.
    - **SNR (Signal to Noise Ratio, snr_target)**: 20.

```{code-cell} ipython3
fid = pyAMARES.kernel.fid.simulate_fid(
    params,
    MHz=120.0,
    sw=10000.0,
    deadtime=200e-6,
    fid_len=1024,
    snr_target=20,
    preview=True,
)
```

## Simple Tutorial on pyAMARES Fitting
- **Initialize the FID Object**: 

```{code-cell} ipython3
FIDobj = pyAMARES.initialize_FID(
    fid=fid,
    MHz=120.0,
    sw=10000.0,
    deadtime=200e-6,
    normalize_fid=False,
    priorknowledgefile="example_human_brain_31P_7T.csv",
    preview=False,
)
```

- **A. HSVD Optimization of Initial Parameters (Optional)**: Utilize HSVD to optimize the initial parameters for fitting, if desired.

```{code-cell} ipython3
params_hsvd = pyAMARES.HSVDinitializer(
    fid_parameters=FIDobj,
    num_of_component=12,  # If an error happens with preview, decrease this number
    fitting_parameters=FIDobj.initialParams,
    preview=False,
)
```

- **Fitting AMARES Using HSVD-Initialized Parameters**:

```{code-cell} ipython3
FIDresult1 = pyAMARES.fitAMARES(
    fid_parameters=FIDobj,
    fitting_parameters=params_hsvd,
    method="least_squares",
    ifplot=True,
)
```

```{code-cell} ipython3
FIDresult1.styled_df
```

- **B. Initialization Using Levenberg-Marquardt Method**: Instead of using the HSVD initializer, initialize the parameters using the Levenberg-Marquardt method.

```{code-cell} ipython3
params_LM = pyAMARES.fitAMARES(
    fid_parameters=FIDobj,
    fitting_parameters=FIDobj.initialParams,
    method="leastsq",
    ifplot=True,
)
```

- **Fitting AMARES Using Levenberg-Marquardt-Initialized Parameters**:

```{code-cell} ipython3
FIDresult2 = pyAMARES.fitAMARES(
    fid_parameters=FIDobj,
    fitting_parameters=params_LM.fittedParams,
    initialize_with_lm=False,
    method="least_squares",
    ifplot=True,
)
```

```{code-cell} ipython3
FIDresult2.styled_df
```

- **New after 0.3.10**: Fitting AMARES using internally initialized Levenberg-Marquardt parameters:

```{code-cell} ipython3
FIDresult2b = pyAMARES.fitAMARES(
    fid_parameters=FIDobj,
    fitting_parameters=FIDobj.initialParams,
    initialize_with_lm=True,  # Turn on the Levenberg-Marquardt initializer
    method="least_squares",
    ifplot=True,
)
```

```{code-cell} ipython3
FIDresult2b.styled_df
```

## Visualize Fitting Results

- **Visualization with pyAMARES**: pyAMARES utilizes the `plotParameter` object to display fitting results visually.
- **Template for `plotParameter`**: Within the initialized `FIDobj`, there is a pre-configured template for `plotParameter` to facilitate customization and usage.

```{code-cell} ipython3
plotParameter = (
    FIDobj.plotParameters
)  # plotParameter is a pointer to FIDobj.plotParameters
# If you do not want to modify FIDobj.plotParameters,
# do plotParameter = deepcopy(FIDobj.plotParameters) instead
plotParameter
```

- **Modify the visualization parameters**

```{code-cell} ipython3
plotParameter.ifphase = True  # Phasing the spectrum for visualization
plotParameter.xlim = (10, -20)  # Show 10 to -20 ppm only
```

```{code-cell} ipython3
pyAMARES.plotAMARES(FIDresult2, plotParameters=plotParameter)
```

- **Uniform Phase for All Peaks**: Previously, each peak's phase was fitted independently. We can now attempt to use the same phase for all peaks.
- **Editing Parameters**: Parameters can be edited programmatically using Python. Alternatively, you can manually edit them using Excel or similar software. Use the `pyAMARES.kernel.lmfit.save_parameter_to_csv` function to save parameters to a CSV file, and `pyAMARES.kernel.lmfit.load_parameter_from_csv` to reload them as an lmfit parameter object.
``

```{code-cell} ipython3
initial_params_fixedphase = deepcopy(
    params_LM.fittedParams
)  # Starting from the Levenberg-Marquardt-Initialized Parameters
```

```{code-cell} ipython3
# Constrain all phase parameters (starting with `phi` ) to the phase of PCr (`phi_Pcr`)
for peak_para in initial_params_fixedphase:
    if peak_para.startswith("phi"):
        initial_params_fixedphase[peak_para].expr = "phi_PCr"
```

```{code-cell} ipython3
# But do not fix phi_PCr itself because it will be fitted
initial_params_fixedphase["phi_PCr"].expr = None
initial_params_fixedphase["phi_PCr"].vary = True
```

- If you modified the `FIDobj.plotParameters` above and turned on `ifphase`, the following preview will show phased spectrum

```{code-cell} ipython3
FIDresult3 = pyAMARES.fitAMARES(
    fid_parameters=FIDobj,
    fitting_parameters=initial_params_fixedphase,
    method="least_squares",
    ifplot=True,
)
```

```{code-cell} ipython3
FIDresult3.styled_df
```

- **Convert lmfit Parameter to Pandas DataFrame**:
    - For comparison and easier editing, import functions that enable conversion between an lmfit Parameter object and a Python pandas DataFrame.

```{code-cell} ipython3
from pyAMARES import parameters_to_dataframe
```

```{code-cell} ipython3
origin = parameters_to_dataframe(params)  # Original parameters
result1 = parameters_to_dataframe(
    FIDresult1.fittedParams
)  # Fitting Result using HSVD initialized parameters
result2 = parameters_to_dataframe(
    FIDresult2.fittedParams
)  # Fitting Result using Levenberg-Marquardt initialized parameters
result3 = parameters_to_dataframe(
    FIDresult3.fittedParams
)  # Fitting Result using fixed phase of all peaks
```

```{code-cell} ipython3
# Generate index for peak amplitudes only.
amplitude_index = origin.name.str.startswith("ak")
amplitude_index
```

```{code-cell} ipython3
# Define a function to do linear regression between two lists
import matplotlib.pyplot as plt
import scipy


def compare_plot(x, y, labellist, title="", xlabel="", ylabel=""):
    assert len(x) == len(y) == len(labellist)
    x = x / x[0]
    y = y / y[0]
    plt.scatter(x, y)
    for i, j, l in zip(x, y, labellist):
        plt.annotate(l, (i * 1.02, j * 1.02))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, "r", label="slope=%.3f" % slope)

    combined_min = min(min(x), min(y)) * 0.95
    combined_max = max(max(x), max(y)) * 1.05
    plt.plot(
        [combined_min, combined_max], [combined_min, combined_max], "k--"
    )  # Dashed diagonal line

    # Beautify the plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.axis('equal')  # Use the same scale for both x and y axes

    # Print the results
    print("Slope: %.3f" % slope)
    print("Pearson's R: %.4f" % r_value)
    print("P-value: %.2e" % p_value)
    plt.title("%s r_value=%.2f p_value=%.2f" % (title, r_value, p_value))
    plt.xlim(combined_min, combined_max)
    plt.ylim(combined_min, combined_max)
    plt.legend()
    # Display the plot
    plt.show()

    # Return the slope, Pearson's R, and p-value
    return slope, r_value, p_value
```

- Now we have three fitting results:
    - `result1`: Fitting result using HSVD-initialized parameters.
    - `result2`: Fitting result using Levenberg-Marquardt-initialized parameters.
    - `result3`: Fitting result with a fixed phase for all peaks.
- Compare them to the ground truth by linear regressions
    - Compare these fitting results to the ground truth using linear regression analyses.

```{code-cell} ipython3
compare_plot(
    y=origin[amplitude_index]["value"],
    x=result1[amplitude_index]["value"],
    labellist=FIDresult1.peaklist,
    xlabel="Fitted Result",
    ylabel="Ground Truth",
)
```

```{code-cell} ipython3
compare_plot(
    x=result2[amplitude_index]["value"],
    y=origin[amplitude_index]["value"],
    labellist=FIDresult1.peaklist,
    xlabel="Fitted Result",
    ylabel="Ground Truth",
)
```

```{code-cell} ipython3
compare_plot(
    x=result3[amplitude_index]["value"],
    y=origin[amplitude_index]["value"],
    labellist=FIDresult1.peaklist,
    xlabel="Fitted Result",
    ylabel="Ground Truth",
)
```

```{code-cell} ipython3

```
