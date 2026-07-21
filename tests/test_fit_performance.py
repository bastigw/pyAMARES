import os

import pandas as pd
import pytest

import pyAMARES
from pyAMARES.util.crlb import _diff_expr_cached

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PK_FILE = os.path.join(CURRENT_DIR, "example_human_brain_31P_7T.csv")
MHZ = 120.0
SW = 10000.0


@pytest.fixture(scope="module")
def fid_parameters():
    """
    A multiplet, expr-constrained prior-knowledge fixture (unlike singlet.csv
    used elsewhere), so create_pmatrix()'s sympy differentiation path is
    actually exercised.
    """
    params, _ = pyAMARES.generateparameter(PK_FILE, MHz=MHZ)
    fid = pyAMARES.simulate_fid(
        params, MHz=MHZ, sw=SW, deadtime=0.0, fid_len=2048, snr_target=100
    )
    return pyAMARES.initialize_FID(
        fid=fid, priorknowledgefile=PK_FILE, MHz=MHZ, sw=SW, deadtime=0.0, preview=False
    )


def test_build_styled_report_false_skips_styler_but_keeps_data(fid_parameters):
    out_styled = pyAMARES.fitAMARES(
        fid_parameters=fid_parameters,
        fitting_parameters=fid_parameters.initialParams,
        method="leastsq",
        ifplot=False,
        build_styled_report=True,
    )
    out_plain = pyAMARES.fitAMARES(
        fid_parameters=fid_parameters,
        fitting_parameters=fid_parameters.initialParams,
        method="leastsq",
        ifplot=False,
        build_styled_report=False,
    )

    # The numeric results must not depend on whether a Styler was built.
    pd.testing.assert_frame_equal(
        out_styled.result_multiplets, out_plain.result_multiplets
    )

    from pandas.io.formats.style import Styler

    assert isinstance(out_styled.styled_df, Styler)
    # styled_df/simple_df are still set (never None) when styling is skipped,
    # just as plain DataFrames instead of Stylers.
    assert isinstance(out_plain.styled_df, pd.DataFrame)
    assert out_plain.simple_df is not None


def test_diff_expr_cached_matches_uncached_sympy():
    import sympy
    from sympy.parsing import sympy_parser

    expr = "ak_Peak_A * 2"
    cached = _diff_expr_cached(expr)
    direct = float(sympy.diff(sympy_parser.parse_expr(expr)).evalf())
    assert cached == pytest.approx(direct)
    # Cache hit returns the identical (memoized) float, not just an equal one.
    assert _diff_expr_cached(expr) is cached
