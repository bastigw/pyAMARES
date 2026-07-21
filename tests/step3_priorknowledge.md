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

# Prior Knowledge Spreadsheet for pyAMARES
**[Try this tutorial on Google Colab!](https://colab.research.google.com/drive/1mVx7avSBsynBnYk_VVMeJWsAgfq4iR0G)**



- PyAMARES imports prior knowledge from spreadsheets to use as initial values and constraints for fitting MRS data based on the AMARES model function.
- The prior knowledge spreadsheet can be in CSV or MS Excel (xlsx) format.
- In this spreadsheet:
    - The upper half defines the initial values.
    - The lower half specifies the fitting constraints.
    - The parameters for a given peak are defined in a column of the spreadsheet.
- `pyAMARES.initialize_FID` can be used to load and preview the prior knowledge spreadsheet.
- **Comments**: Lines starting with `#` can be used to add comments to the prior knowledge spreadsheet. In the CSV format, comments cannot be added to the first rows. However, this limitation does not apply to the Excel (xlsx) format.

+++

- A simple example of peak parameter of a singlet

```{code-cell} ipython3
single_obj = pyAMARES.initialize_FID(
    fid=None, priorknowledgefile="singlet.csv", preview=True
)
```

- **Initial Fitting Parameter**
    - The parsed initial prior knowledge is converted into an lmfit Parameter object.
    - Each parameter includes initial values, minimum and maximum limits, and a `vary` flag that indicates whether it is fixed during the fitting process.
    - In addition to editing the input spreadsheet, fitting parameters can also be manually modified in the code.
    - **Refinement of Fitted Parameters**: The fitted parameters (not shown in this tutorial) will be returned in the same format and can be refined for subsequent rounds of fitting.

+++

## Spreadsheet Format

- **Index Column**: Always use the terms `amplitude`, `chemicalshift`, `linewidth`, `phase`, and `g` as index labels in the spreadsheet for both initial values and constraints  

- **Setup Constraints**:
   - Constraints are set using brackets. For example, `(-180, 180)` indicates a range from -180 to 180.
   - If only a lower bound is needed, omit the second half of the bracket. For example, `(0,` specifies a range of 0 and above.   
   - **(New after version 0.3.4)** If only a single value is specified in a constraint cell, 
   the corresponding parameter is fixed and will not be fitted.

- **Physical Units**:
    - In the spreadsheet, the `amplitude` and `g` values are unitless. `chemicalshift` is measured in ppm, `linewidth` in Hz, and `phase` in degrees.

+++

## Setting Up J-coupling Splitted Multiplets: An Example of In Vivo 31P MRS of the Human Brain at 7T

- **Peak Name Suffix**:
    - To set up a multiplet, designate the main sublet peak using ASCII letters, and define other sublets by adding numeric suffixes to the main peak name. For instance, the triplet for $\beta$-ATP is labeled `BATP`, `BATP2`, and `BATP3`.
    - Therefore, the numbers are not allowed in other peak names. 
    - Similarly, the doublet for $\gamma$-ATP is labeled `GATP` and `GATP2`.

- **Constraints for Multiplets**:
    - Parameters can be constrained using mathematical expressions, which is especially useful for multiplet setups.
    - Multiplets separated by J-coupling share parameters like phase and linewidth (LW). Constraints for these can be linked to the main peak name; for example, `BATP` in the `LW` and `phase` rows.
    - The `chemicalshifts` of sublets can be constrained relative to the main peak using its peak name and the J-coupling constants. For example, `BATP-15Hz` indicates the `chemicalshift` is set 15 Hz lower than that of $\beta$-ATP. If ppm is used, it will be converted to Hz using the `MHz` argument.
    - The `amplitude` of sublets can be related to the main peak. For instance, with $\beta$-ATP as a triplet having 1:2:1 amplitude ratios, the amplitude constraints for the sublets could be set as `BATP/2`. Similarly, for $\gamma$-ATP, where two sublets have a 1:1 amplitude ratio, the amplitude can be set as `GATP`.
    - Since the prior knowledge dataset spreadsheet is parsed from left to right, the peak that will be mathematically constrained to it must always be put to the left of the peaks that will be constrained. For example, for the multiplets, the main peak, such as `BATP`, will always be put to the left of `BATP2` or `BATP3`, whose amplitude constraints will be fixed as `BATP/2`.


```{code-cell} ipython3
multiplet_obj = pyAMARES.initialize_FID(
    fid=None, priorknowledgefile="example_human_brain_31P_7T.csv", preview=True
)
```

- **Initial Fitting Parameter**

```{code-cell} ipython3
multiplet_obj.initialParams
```

## Use a Single Value in the Constraint Cell to Fix the Corresponding Parameter (New after version 0.3.4)

```{code-cell} ipython3
FixExampleObj = pyAMARES.initialize_FID(
    fid=None, priorknowledgefile="Table1.csv", preview=True
)
```

- In this prior knowledge spreadsheet, the phases are fixed at either 0 or 180, as determined by the single-value constraint cells. As shown in the initial fitting parameters below, all phase parameters are fixed (`vary=False`) and will not be fitted.

```{code-cell} ipython3
loadedpk_pd = pyAMARES.parameters_to_dataframe(FixExampleObj.initialParams)
loadedpk_pd.loc[loadedpk_pd.name.str.startswith("phi")]
```

```{code-cell} ipython3

```
