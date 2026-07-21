import os

import pytest
from lmfit.minimizer import MinimizerResult

import pyAMARES
from pyAMARES.kernel.lmfit import AMARESFitSummary

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PK_FILE = os.path.join(CURRENT_DIR, "singlet.csv")
MHZ = 120.0
SW = 10000.0


@pytest.fixture(scope="module")
def fid_parameters():
    """A small, fast, synthetic FID + prior knowledge fixture for fitAMARES tests."""
    params, _ = pyAMARES.generateparameter(PK_FILE, MHz=MHZ)
    fid = pyAMARES.simulate_fid(
        params, MHz=MHZ, sw=SW, deadtime=0.0, fid_len=2048, snr_target=100
    )
    return pyAMARES.initialize_FID(
        fid=fid, priorknowledgefile=PK_FILE, MHz=MHZ, sw=SW, deadtime=0.0, preview=False
    )


def test_fitAMARES_returns_namespace_with_typed_summary(fid_parameters):
    out = pyAMARES.fitAMARES(
        fid_parameters=fid_parameters,
        fitting_parameters=fid_parameters.initialParams,
        method="leastsq",
        ifplot=False,
    )
    # Legacy attributes must still be present (non-breaking change).
    assert hasattr(out, "fittedParams")
    assert hasattr(out, "out_obj")
    assert hasattr(out, "resNormSq")
    assert hasattr(out, "relativeNorm")

    assert isinstance(out.fit_summary, AMARESFitSummary)
    assert out.fit_summary.method == "leastsq"
    assert isinstance(out.fit_summary.success, bool)
    assert out.fit_summary.nfev > 0
    assert out.fit_summary.redchi == pytest.approx(out.out_obj.redchi)
    assert out.fit_summary.resNormSq == pytest.approx(out.resNormSq)
    assert out.fit_summary.relativeNorm == pytest.approx(out.relativeNorm)
    assert out.fit_summary.elapsed_seconds > 0


def test_fitAMARES_inplace_returns_minimizer_result_with_summary(fid_parameters):
    out = pyAMARES.fitAMARES(
        fid_parameters=fid_parameters,
        fitting_parameters=fid_parameters.initialParams,
        method="leastsq",
        ifplot=False,
        inplace=True,
    )
    assert isinstance(out, MinimizerResult)
    assert isinstance(out.fit_summary, AMARESFitSummary)
    assert out.fit_summary.method == "leastsq"


def test_amares_fit_summary_str_is_descriptive():
    summary = AMARESFitSummary(
        method="leastsq",
        success=True,
        message="Fit succeeded.",
        nfev=42,
        redchi=0.001,
        resNormSq=1.23,
        relativeNorm=0.045,
        elapsed_seconds=0.5,
    )
    text = str(summary)
    assert "leastsq" in text
    assert "converged" in text
    assert "42" in text
