from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime
from functools import partial

import pandas as pd
from loguru import logger

from ..kernel.lmfit import fitAMARES


def fit_dataset(
    fid_current,
    FIDobj_shared,
    initial_params,
    method="leastsq",
    initialize_with_lm=False,
    objective_func=None,
    return_out=False,
    build_styled_report=False,
):
    """
    Fits a dataset to a shared FID Parameter object using the AMARES algorithm
    with specified initial parameters and fitting method.

    This function deep copies a shared FID object so that it won't conflict in
    the multiprocessing. Then it updates its FID with the current dataset, and
    applies the AMARES fitting algorithm using the provided initial parameters.
    The fitting results are returned in the pandas dataframe ``FIDobj.result_multiplet``

    Args:
        fid_current (array-like): The current FID dataset to be fitted.
        FIDobj_shared (FID object): A shared FID object template to be used for fitting. This object should contain common settings and parameters applicable to all datasets.
        initial_params (lmfit.Parameters): Initial fitting parameters for the AMARES algorithm.
        method (str, optional): The fitting method to be used. Defaults to "leastsq" (Levenberg-Marquardt).
        initialize_with_lm (bool, optional, default False, new in 0.3.9): If True, a Levenberg-Marquardt initializer (``least_sq``) is executed internally. See ``pyAMARES.lmfit.fitAMARES`` for details.
        objective_func (callable, optional): Custom objective function for ``pyAMARES.lmfit.fitAMARES``. If None,
          the default objective function will be used. Defaults to None.
        build_styled_report (bool, optional): Forwarded to ``fitAMARES``/``report_amares``.
          Defaults to False here (unlike ``fitAMARES``'s own default of True), since building
          the CRLB-highlighted ``pandas.Styler`` display table is the single most expensive
          part of fitting and its output is essentially never used when fitting datasets one
          voxel at a time in a batch job — only the numeric ``result_multiplets`` table
          returned by this function is. Pass True if you specifically need ``out.styled_df``.


    Returns:
        pandas.DataFrame or None: A DataFrame containing the fitting results for the current dataset. Returns None if an error occurs during fitting.

    Raises:
        Exception: If an error occurs during the fitting process, it is caught and a message is printed to the console, and None is returned.
    """
    try:
        FIDobj_current = deepcopy(FIDobj_shared)
        FIDobj_current.fid = fid_current
        if objective_func is None:
            out = fitAMARES(
                fid_parameters=FIDobj_current,
                fitting_parameters=initial_params,
                method=method,  # Use method passed as a parameter to the function
                initialize_with_lm=initialize_with_lm,  # New in 0.3.9
                ifplot=False,
                build_styled_report=build_styled_report,
            )
        else:
            out = fitAMARES(
                fid_parameters=FIDobj_current,
                fitting_parameters=initial_params,
                method=method,  # Use method passed as a parameter to the function
                initialize_with_lm=initialize_with_lm,  # New in 0.3.9
                ifplot=False,
                objective_func=objective_func,
                build_styled_report=build_styled_report,
            )

        result_table = out.result_multiplets
        del FIDobj_current
        if return_out:
            return result_table, out
        else:
            del out
            return result_table
    except Exception as e:
        logger.critical(f"Error in fit_dataset: {e}")
        return None


def run_parallel_fitting_with_progress(
    fid_arrs,
    FIDobj_shared,
    initial_params,
    method="leastsq",
    initialize_with_lm=False,
    num_workers=8,
    logfilename="logs/parellelfitting.log",
    loglevel=31,
    objective_func=None,
    notebook=True,
    chunksize=1,
):
    """
    Runs parallel AMARES fitting of multiple FID datasets using a shared FID object template and initial parameters.

    This function deep copies a shared FID object and performs parallel fitting on an array of FID datasets.
    It utilizes a process pool to handle the fitting tasks concurrently, logging progress and results to a
    specified file ``logfilename``. The execution time is printed upon completion.

    Args:
        fid_arrs (numpy.ndarray): An array of FID datasets to be fitted, where
          each row corresponds to a different dataset.
        FIDobj_shared (FID object): A shared FID object template to be used
          for all fitting tasks. This object should contain common settings and
          parameters applicable to all datasets.
        initial_params (lmfit.Parameters): Initial fitting parameters for the AMARES algorithm.
        method (str, optional): The fitting method to be used. Defaults to 'leastsq' (Levenberg-Marquardt).
        initialize_with_lm (bool, optional, default False, new in 0.3.9):
          If True, a Levenberg-Marquardt initializer (``least_sq``) is executed internally. See ``pyAMARES.lmfit.fitAMARES`` for details.
        num_workers (int, optional): The number of worker processes to use in parallel processing. Defaults to 8.
        logfilename (str, optional): The name of the file where the progress log is saved. Defaults to 'logs/parellelfitting.log'.
        loglevel (int, optional): The logging level for the logger. Defaults to 31 - just above warning.
        objective_func (callable, optional): Custom objective function for ``pyAMARES.lmfit.fitAMARES``. If None,
          the default objective function will be used. Defaults to None.
        notebook (bool, optional): If True, uses tqdm.notebook for progress display in Jupyter notebooks.
          If False, uses standard tqdm. Defaults to True.
        chunksize (int, optional): Number of datasets dispatched to a worker per IPC
          round-trip (see ``concurrent.futures.Executor.map``). Defaults to 1 (one
          dataset per task, matching prior behavior). Increasing it amortizes
          per-task scheduling overhead across more fits per round-trip; worth raising
          for very large batches of individually fast fits.

    Returns:
        list: A list of fitting result objects (e.g., pandas DataFrames) for each FID dataset.
    """
    if notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    FIDobj_shared = deepcopy(FIDobj_shared)
    try:
        del FIDobj_shared.styled_df
    except AttributeError:
        logger.warning("There is no styled_df!")
    try:
        del FIDobj_shared.simple_df
    except AttributeError:
        logger.warning("There is no simple_df!")
    timebefore = datetime.now()
    results = []

    loggerID = logger.add(logfilename, level=loglevel, rotation="10 min")
    try:
        logger.level("BATCH_INFO", no=loglevel)
    except ValueError:
        # This means the logger level "BATCH_INFO" was already added
        pass

    data = [
        {
            "name": name,
            "value": float(par.value),
            "min": par.min,
            "max": par.max,
            "vary": par.vary,
            "expr": par.expr,
            "brute_step": par.brute_step,
            "stderr": par.stderr,
        }
        for name, par in initial_params.items()
    ]

    df = pd.DataFrame(data)
    df.set_index("name", inplace=True)
    df.sort_values(by="name", inplace=True)
    logger.log(
        "BATCH_INFO", f"Initial Paramerters used for batch fitting:\n{df.to_string()}"
    )

    logger.log(
        "BATCH_INFO",
        f"Starting fitting of {len(fid_arrs)} datasets with parallel processing. Number of workers: {num_workers}",
    )
    # executor.map streams work to the pool instead of eagerly creating one
    # Future per dataset up front, which matters for large MRSI-style batches
    # (tens of thousands of voxels). chunksize batches multiple datasets per
    # IPC round-trip; the default of 1 preserves the previous per-task behavior.
    fit_one = partial(
        fit_dataset,
        FIDobj_shared=FIDobj_shared,
        initial_params=initial_params,
        method=method,
        initialize_with_lm=initialize_with_lm,
        objective_func=objective_func,
    )
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        result_iter = executor.map(
            fit_one,
            (fid_arrs[i, :] for i in range(fid_arrs.shape[0])),
            chunksize=chunksize,
        )
        for result in tqdm(
            result_iter, total=fid_arrs.shape[0], desc="Processing Datasets"
        ):
            results.append(result)

    logger.log(
        "BATCH_INFO",
        "Fitting completed. If no errors were logged, all fits were successful.",
    )

    timeafter = datetime.now()
    logger.log(
        "BATCH_INFO",
        f"Fitting {len(fid_arrs)} spectra with {num_workers} processors took {(timeafter - timebefore).total_seconds()} seconds",
    )

    logger.remove(loggerID)
    return results
