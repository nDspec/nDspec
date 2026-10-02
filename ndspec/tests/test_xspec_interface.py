import ctypes as ct
import os
import re
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pytest

import ndspec.XspecInterface as X

LIBEXT = ".dylib" if sys.platform == "darwin" else ".so"
MOCK = "mock"

MODEL_DAT = """\
nthComp        5  0.         1.e20          donthcomp  add  0
Gamma      " "    1.7   1.001   1.005   5.     10.     0.01
kT_e       keV    100.  1.      5.      1000.  1000.   0.1
kT_bb      keV    0.1   1.e-3   1.e-2   10.    10.     -0.01
$inp_type  " "    0.
Redshift   " "    0.    -0.999  -0.999  10.    10.     -0.01

gaussian       2  0.         1.e20          C_gaussianLine  add  0
LineE      keV     6.5   0.      0.      1.e6   1.e6    0.05
Sigma      keV     0.1   0.      0.      10.    20.     0.05

TBabs          1  0.         1.e20          C_tbabs   mul  0
nH         "10^22 atoms cm^-2"  1.   0.      0.      1.e5   1.e6    1.e-3
ismabs         1  0.         1.e20          F_ismabs   mul  0
H    10^22   0.1  0 0 1e5 1e6 1e-3


bwcycl         2  0.  1.e20  c_beckerwolff  add  0
*Radius   km  10

M     " "   1.4   1 1 3 3 0.01
blbd_like      1  0.  1.e20  xsblbd  add  0
kT  keV 3.0 1e-3 1e-2 100 200 0.01
cfall          1  0.  1.e20  c_cfall  mul  0
x  " " 1.0 0 0 10 10 0.01
prec4          1  0.  1.e20  prec4  mul  0
x  " " 0.5 -10 -10 10 10 0.01
prec8          1  0.  1.e20  F_prec8  mul  0
x  " " 0.5 -10 -10 10 10 0.01
precc          1  0.  1.e20  C_precc  mul  0
x  " " 0.5 -10 -10 10 10 0.01
gsmooth   2  0. 1.e20  C_gsmooth  con 0
Sig_6keV keV 1.0 0 0 10 20 0.05
Index " " 0 -1 -1 1 1 -0.01 P
relxcpp   1  0. 1.e20  C_relxcpp  add 0
a " " 0.9 -1 -1 1 1 0.01
mixmod    1  0. 1.e20  C_mixmod  mix 0
a " " 0.9 -1 -1 1 1 0.01
"""

MOCK_SRC = r"""
#include <string.h>

static int calls = 0, last_ne = -1, last_spec = -1;
static char last_init[64] = "unset";

#define CSIG const double* e, int n, const double* p, int s, \
             double* f, double* fe, const char* init
#define RECORD_C  calls++; last_ne = n; last_spec = s; \
                  strncpy(last_init, init ? init : "(null)", 63);
#define RECORD_F  calls++; last_ne = *n; last_spec = *ifl;

int mock_calls(void)        { return calls; }
int mock_last_ne(void)      { return last_ne; }
int mock_last_spec(void)    { return last_spec; }
const char* mock_last_init(void) { return last_init; }

/* Fortran, single precision: flux = Gamma * bin width */
void donthcomp_(const float* e, const int* n, const float* p, const int* ifl,
                float* f, float* fe)
{ RECORD_F for (int i = 0; i < *n; i++) f[i] = p[0] * (e[i+1] - e[i]); }

/* Fortran, double precision, with HEASOFT-style C_ wrapper: flux = H */
void C_ismabs(CSIG) { RECORD_C for (int i = 0; i < n; i++) f[i] = p[0]; }
void ismabs_(const double* e, const int* n, const double* p, const int* ifl,
             double* f, double* fe)
{ RECORD_F for (int i = 0; i < *n; i++) f[i] = p[0]; }

/* precision probes: return the first parameter unchanged */
void prec4_(const float* e, const int* n, const float* p, const int* ifl,
            float* f, float* fe)
{ RECORD_F for (int i = 0; i < *n; i++) f[i] = p[0]; }
void prec8_(const double* e, const int* n, const double* p, const int* ifl,
            double* f, double* fe)
{ RECORD_F for (int i = 0; i < *n; i++) f[i] = p[0]; }
void C_precc(CSIG) { RECORD_C for (int i = 0; i < n; i++) f[i] = p[0]; }

/* C++-style (C_ wrapper): flux = LineE * bin width */
void C_gaussianLine(CSIG) { RECORD_C for (int i = 0; i < n; i++) f[i] = p[0] * (e[i+1] - e[i]); }

/* multiplicative: flux = nH */
void C_tbabs(CSIG) { RECORD_C for (int i = 0; i < n; i++) f[i] = p[0]; }

/* C-style (c_ prefix, symbol without prefix): flux = M * bin width */
void beckerwolff(CSIG) { RECORD_C for (int i = 0; i < n; i++) f[i] = p[1] * (e[i+1] - e[i]); }

/* fallbacks: only the C_ wrapper exists for these */
void C_xsblbd(CSIG) { RECORD_C for (int i = 0; i < n; i++) f[i] = 3.0 * (e[i+1] - e[i]); }
void C_cfall(CSIG)  { RECORD_C for (int i = 0; i < n; i++) f[i] = 7.0; }

/* convolution: scales the input spectrum in place */
void C_gsmooth(CSIG) { RECORD_C for (int i = 0; i < n; i++) f[i] *= p[0]; }
"""

F90_SRC = """\
subroutine donthcomp(ear, ne, param, ifl, photar, photer)
  integer ne, ifl
  real*4 ear(0:ne), param(*), photar(ne), photer(ne)
  integer i
  do i = 1, ne
     photar(i) = param(1) * (ear(i) - ear(i-1)) + real(ifl - 1)
  end do
end subroutine
subroutine ismabs(ear, ne, param, ifl, photar, photer)
  integer ne, ifl
  real*8 ear(0:ne), param(*), photar(ne), photer(ne)
  integer i
  do i = 1, ne
     photar(i) = param(1) + dble(ne)
  end do
end subroutine
"""


def _c_compiler():
    for name in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if name and shutil.which(name):
            return shutil.which(name)
    return None


def _build(source, path, compiler=None, suffix=".c"):
    compiler = compiler or _c_compiler()
    if compiler is None:
        pytest.skip("no C compiler available to build the mock model library")
    srcfile = str(path) + suffix
    with open(srcfile, "w") as fh:
        fh.write(source)
    subprocess.check_call([compiler, "-shared", "-fPIC", "-O0", "-o", str(path), srcfile])
    return str(path)


def _write(tmp_path, text, name="model.dat"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


@pytest.fixture(scope="module")
def mock_paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("xsmock")
    # Never name a mock after a real HEASOFT library: on macOS, dlopen() looks
    # up the file name in $DYLD_LIBRARY_PATH before the path it was given.
    lib = _build(MOCK_SRC, d / ("libndspec_xsmock" + LIBEXT))
    return lib, _write(d, MODEL_DAT, "lmodel.dat")


@pytest.fixture(scope="module")
def lib(mock_paths):
    interface = X.ModelInterface()
    interface.add_library(*mock_paths, name=MOCK)
    return interface


def _getter(interface, name, restype=ct.c_int):
    f = getattr(interface._libs[MOCK], name)
    f.restype, f.argtypes = restype, []
    return f()


# =============================================================================
# model.dat parsing
# =============================================================================
def test_parser(tmp_path):
    """MODEL_DAT has irregular blank lines (none, several, whitespace-only,
    one inside an entry), quoted multi-word units, switch/scale parameters
    and a periodic flag."""
    info = X.parse_model_file(_write(tmp_path, MODEL_DAT))
    assert len(info) == 13
    assert info["nthcomp"]["func_call"] == "donthcomp"      # names lower-cased
    nth = info["nthcomp"]["parameters"]
    assert list(nth) == ["Gamma", "kT_e", "kT_bb", "inp_type", "Redshift", "norm"]
    assert nth["Gamma"] == {"value": 1.7, "min": 1.001, "max": 10.0, "unit": "n/a"}
    assert nth["inp_type"] == {"value": 0.0, "min": None, "max": None, "unit": "n/a"}
    assert list(info["bwcycl"]["parameters"]) == ["Radius", "M", "norm"]
    assert info["tbabs"]["parameters"]["nH"]["unit"] == "10^22 atoms cm^-2"
    assert info["gsmooth"]["parameters"]["Index"]["max"] == 1.0
    for name, entry in info.items():
        assert ("norm" in entry["parameters"]) == (entry["type"] == "add"), name


def test_parser_bad_input(tmp_path):
    """A malformed parameter warns and is skipped; a truncated last entry keeps
    what it has; neither affects the other models."""
    text = ("aa 2 0. 1e20 aa mul 0\n"
            "p keV not a number\n"
            "q keV 1 0 0 10 10 0.1\n"
            "bb 3 0. 1e20 bb mul 0\n"
            "r keV 1 0 0 10 10 0.1\n")
    with pytest.warns(UserWarning, match="aa"):
        info = X.parse_model_file(_write(tmp_path, text))
    assert list(info["aa"]["parameters"]) == ["q"]
    assert list(info["bb"]["parameters"]) == ["r"]


# =============================================================================
# Libraries
# =============================================================================
def test_add_library(lib, mock_paths):
    assert set(lib.libraries) == {"xspec", MOCK}
    assert lib.libraries[MOCK]["pars_path"] == mock_paths[1]
    # names shared with Xspec are listed under both libraries
    assert lib.available_models()["gaussian"] == ["xspec", MOCK]
    assert lib.available_models(MOCK)["prec4"] == [MOCK]
    with pytest.raises(ValueError, match="already loaded"):
        lib.add_library(*mock_paths, name=MOCK)


def test_default_library_name(mock_paths):
    interface = X.ModelInterface()
    assert interface.add_library(*mock_paths) == "ndspec_xsmock"


def test_name_clash_requires_library(lib):
    with pytest.raises(ValueError, match="several libraries.*library="):
        lib.add_model("gaussian")
    lib.add_model("gaussian", library=MOCK)
    assert lib.models_info["gaussian"]["library"] == MOCK
    lib.add_model("gaussian", library="xspec")
    assert lib.models_info["gaussian"]["library"] == "xspec"
    lib.add_model("prec4")                       # only defined once: no library needed
    assert lib.models_info["prec4"]["library"] == MOCK


def test_add_model_errors(lib):
    with pytest.raises(KeyError, match="notamodel"):
        lib.add_model("notamodel")
    with pytest.raises(KeyError, match="No library"):
        lib.add_model("tbabs", library="nope")
    with pytest.raises(NotImplementedError, match="mix"):
        lib.add_model("mixmod", library=MOCK)
    with pytest.raises(AttributeError, match="C_relxcpp"):
        lib.add_model("relxcpp", library=MOCK)
    assert not hasattr(lib, "relxcpp")
    # blbd_like and cfall only have a C_ wrapper, which is not used as a fallback
    assert lib.check_models() == {MOCK: ["blbd_like", "cfall", "relxcpp"]}


def test_double_precision_fortran(lib):
    """F_ models use their double-precision Fortran routine, or HEASOFT's C_
    wrapper if that is all the library has."""
    third = 1.0 / 3.0
    assert np.all(lib.add_model("prec8")(EAR, [third]) == third)               # prec8_ only
    assert np.all(lib.add_model("ismabs", library=MOCK)(EAR, [third]) == third)


def test_deprecated_names(mock_paths):
    """FortranInterface/CInterface still work, including the old dummy-function
    add_model, and look in the library they were created with."""
    def gaussian(ear, params):
        pass

    with pytest.warns(DeprecationWarning):
        old = X.CInterface(*mock_paths)
    old.load_models({"gaussian": gaussian})
    assert old.models_info["gaussian"]["library"] == "ndspec_xsmock"
    with pytest.warns(DeprecationWarning):
        old = X.FortranInterface()
    old.add_model(gaussian)
    assert old.models_info["gaussian"]["library"] == "xspec"

EAR = np.array([1.0, 1.5, 2.5, 4.0, 8.0])   # non-uniform bins


@pytest.fixture
def loaded(lib):
    lib.load_models(["nthcomp", "gaussian", "tbabs", "bwcycl", "prec4", "precc", "gsmooth"],
                    library=MOCK)
    return lib


@pytest.mark.parametrize("model, params, k", [
    ("nthcomp", [2.0, 100, 0.1, 0, 0], 2.0),    # Fortran, single precision
    ("gaussian", [6.5, 0.1], 6.5),              # C_ wrapper
    ("bwcycl", [10, 1.4], 1.4),                 # c_ style
])
def test_additive(loaded, model, params, k):
    """Mocks return flux = k * bin width per bin; nDspec returns flux per keV
    times the norm, which it applies itself."""
    for norm in (0.0, 2.5):
        out = getattr(loaded, model)(EAR, params + [norm])
        assert out.shape == (EAR.size - 1,) and out.dtype == np.float64
        assert np.allclose(out, k * norm)


def test_multiplicative(loaded):
    assert np.allclose(loaded.tbabs(EAR, [0.25]), 0.25)   # no width division, no norm


def test_convolution(loaded):
    seed = np.array([1.0, 2.0, 3.0, 4.0])
    out = loaded.gsmooth(EAR, [2.0, 0.0], seed)
    assert np.allclose(out, [2.0, 4.0, 6.0, 8.0])
    assert np.array_equal(seed, [1.0, 2.0, 3.0, 4.0])   # input untouched
    with pytest.raises(ValueError, match="seed"):
        loaded.gsmooth(EAR, [2.0, 0.0], np.ones(EAR.size))


@pytest.mark.parametrize("model, dtype", [("prec4", np.float32), ("precc", np.float64)])
def test_precision(loaded, model, dtype):
    third = 1.0 / 3.0
    assert np.all(getattr(loaded, model)(EAR, [third]) == np.float64(dtype(third)))


@pytest.mark.parametrize("model, params", [("gaussian", [6.5, 0.1, 1.0]),        # C
                                           ("nthcomp", [2.0, 100, 0.1, 0, 0, 1.0])])  # Fortran
def test_arguments_received(loaded, model, params):
    getattr(loaded, model)(EAR, params)
    assert _getter(loaded, "mock_last_ne") == EAR.size - 1
    assert _getter(loaded, "mock_last_spec") == 1
    if model == "gaussian":
        assert _getter(loaded, "mock_last_init", ct.c_char_p) == b""


def test_input_types_and_layouts(loaded):
    """Inputs are converted to contiguous arrays of the right dtype."""
    ref = loaded.gaussian(EAR, [6.5, 0.1, 1.0])
    # lists, tuples and ints are accepted
    assert np.allclose(loaded.gaussian(list(EAR), (6.5, 0.1, 1)), ref)
    # non-contiguous slice: without a copy the library would read the -99 fillers
    big = np.full(2 * EAR.size, -99.0)
    big[::2] = EAR
    assert np.allclose(loaded.gaussian(big[::2], [6.5, 0.1, 1.0]), ref)            # C
    assert np.allclose(loaded.nthcomp(big[::2], [2.0, 100, 0.1, 0, 0, 1.0]), 2.0)  # Fortran
    # a single edge gives no bins
    with pytest.raises(ValueError, match="ear"):
        loaded.gaussian([1.0], [6.5, 0.1, 1.0])


# =============================================================================
# Parameter checks
# =============================================================================
@pytest.mark.parametrize("params, bad", [
    ([0.5, 100, 0.1, 0, 0, 1.0], "Gamma"),        # first parameter
    ([2.0, 100, 0.1, 0, 20.0, 1.0], "Redshift"),  # a later one (old code only checked the first)
    ([2.0, 100, 0.1, 0, 0], "Wrong parameter number"),
])
def test_invalid_parameters_return_nan(loaded, params, bad):
    before = _getter(loaded, "mock_calls")
    with pytest.warns(UserWarning, match=bad):
        out = loaded.nthcomp(EAR, params)
    assert out.shape == (EAR.size - 1,) and np.all(np.isnan(out))
    assert _getter(loaded, "mock_calls") == before      # model not called


def test_valid_edge_parameters(loaded):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # values exactly at the hard limits are allowed, even for float32
        # models (1.001 rounds below 1.001 in float32); switches have no bounds
        loaded.nthcomp(EAR, [1.001, 100, 0.1, 12345.0, -0.999, 1.0])
        loaded.nthcomp(EAR, [10.0, 100, 0.1, 0, 10.0, 1.0])


# =============================================================================
# Real gfortran ABI (optional)
# =============================================================================
@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not available")
def test_real_fortran_subroutine(tmp_path):
    path = tmp_path / ("libndspec_f" + LIBEXT)
    _build(F90_SRC, path, compiler=shutil.which("gfortran"), suffix=".f90")
    dat = _write(tmp_path, "nthComp 1 0. 1e20 donthcomp add 0\n"
                           "Gamma \" \" 1.7 1.001 1.005 5. 10. 0.01\n")
    interface = X.ModelInterface()
    interface.add_library(str(path), dat, name="f90")
    interface.add_model("nthcomp", library="f90")
    # the subroutine adds (ifl - 1): ne and ifl must arrive by reference intact
    assert np.allclose(interface.nthcomp(EAR, [2.0, 1.0]), 2.0)


# =============================================================================
# The real Xspec library (from xspectrampoline, or $HEADAS if set)
# =============================================================================
_HEADER = re.compile(r"^(\S+)\s+(\d+)\s+\S+\s+\S+\s+(\S+)\s+(add|mul|con|mix|acn|amx)\b")


@pytest.fixture(scope="module")
def xspec():
    return X.ModelInterface()


def test_xspec_model_dat_and_symbols(xspec):
    """Every model in the real model.dat parses with the declared number of
    parameters and can be found in the real library."""
    expected = {}
    with open(xspec.libraries["xspec"]["pars_path"]) as fh:
        for line in fh:
            m = _HEADER.match(line)
            if m:
                expected[m.group(1).lower()] = int(m.group(2)) + (m.group(4) == "add")
    models = xspec.libraries["xspec"]["models"]
    assert {k: len(v["parameters"]) for k, v in models.items()} == expected
    assert xspec.check_models() == {}


def test_xspec_models_evaluate(xspec):
    # Only models that need no data files: xspectrampoline does not ship
    # spectral/modelData, and some models (e.g. kerrbb) exit the process
    # when a data file is missing.
    for model in ["powerlaw", "tbabs", "nthcomp", "gaussian", "diskbb"]:
        xspec.add_model(model)
        pars = [v["value"] for v in xspec.models_info[model]["parameters"].values()]
        out = getattr(xspec, model)(np.logspace(-1, 2, 200), pars)
        assert np.all(np.isfinite(out)) and np.any(out != 0), model
