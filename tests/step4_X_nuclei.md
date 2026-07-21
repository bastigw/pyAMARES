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

pyAMARES.__version__
```

# Examples of In Vivo X-Nuclei ($^{129}$Xe and $^{2}$H) MRS Fitting
- Reproduce Figures 2B, C and Figures S2C, D of the [pyAMARES publication](https://doi.org/10.3390/diagnostics14232668)

+++

**[Try this tutorial on Google Colab!](https://colab.research.google.com/drive/1HGFB0G0NuHxpa2lfUAx-_sGkKx7QwVeO)**

+++

## Fitting a Voxel of Hyperpolarized $^{129}$Xe MRSI Acquired from Healthy Porcine Lungs at 3T 
- **Set Scanner Parameters**:
    - **MHz (Field Strength)**: 35.340772 MHz, corresponding to $^{129}$Xe at 3T
    - **sw (Spectral Width)**: 20000 Hz
    - **Deadtime**: 7.14e-05 seconds

```{code-cell} ipython3
MHz = 35.340772
sw = 20000
begin_time = 7.14e-05
```

- **Load the FID of an Example Voxel of** $^{129}$Xe MRSI

```{code-cell} ipython3
fid = pyAMARES.readmrs("a_voxel_Xe.txt")
```

- **Initialize the FID Object**

```{code-cell} ipython3
# Initialize an FIDobj using the loaded fid and spectral parameters
FIDobj = pyAMARES.initialize_FID(
    fid,
    priorknowledgefile="FigS2A.csv",  # Prior knowledge file for hyperpolarized 129Xe
    MHz=MHz,
    sw=sw,
    deadtime=begin_time,
    preview=True,
    g_global=False,
)  # When g_global is False, the lineshape parameter `g` will be fitted based on prior knowledge constraints
```

- Note: In the prior knowledge dataset, **the gas and red blood cell (RBC) signals are modeled with Lorentzian lineshapes (g = 0), while the membrane signal is modeled with a Voigt lineshape (initial value: g = 0.1)** to account for its structural heterogeneity. This approach has become widely accepted in the xenon MRS community for quantifying membrane and RBC signals [Bier et al., NMR Biomed. 2019](https://doi.org/10.1002/nbm.4029)

+++

- **First Round of Fitting: Parameter Optimization**

```{code-cell} ipython3
out1 = pyAMARES.fitAMARES(
    fid_parameters=FIDobj,
    fitting_parameters=FIDobj.initialParams,
    method="leastsq",  # Initialize parameters using the Levenberg-Marquardt method
    ifplot=True,
)
```

- **Optimized Fitting Parameters**

```{code-cell} ipython3
out1.fittedParams
```

- **Fix Lineshape Parameters of Gas and RBC for the AMARES Fitting**

```{code-cell} ipython3
out1.fittedParams["g_Gas"].vary = False
out1.fittedParams["g_RBC"].vary = False
```

```{code-cell} ipython3
out1.fittedParams
```

- **Fitting AMARES Using Levenberg-Marquardt-Initialized Parameters:**

```{code-cell} ipython3
out2 = pyAMARES.fitAMARES(
    fid_parameters=out1,
    fitting_parameters=out1.fittedParams,  # Fit Xenon data using optimized parameters with fixed g_Gas and g_RBC
    method="least_squares",
    ifplot=True,
)
```

- **Visualization of AMARES Fitting as shown in Figure 2B** of [pyAMARES Publication](https://doi.org/10.3390/diagnostics14232668)

```{code-cell} ipython3
# Modify the visualization parameters
plotParameters = out2.plotParameters
plotParameters.xlim = (300, -300)  # Show spectrum from 300 to -300 ppm
plotParameters.ifphase = False  # Do not apply phase correction
plotParameters.lb = 5  # Apply line broadening of 5 Hz for visualization
```

```{code-cell} ipython3
pyAMARES.plotAMARES(out2, plotParameters=plotParameters)
```

- **Obtained Fitting Results Spreadsheet as shown in Figure S2C** of [pyAMARES Publication](https://doi.org/10.3390/diagnostics14232668)

```{code-cell} ipython3
out2.simple_df
```

- **Obtained Fitting Results Spreadsheet as shown in Figure S2D** of [pyAMARES Publication](https://doi.org/10.3390/diagnostics14232668)

```{code-cell} ipython3
out1.simple_df
```

```{code-cell} ipython3

```
