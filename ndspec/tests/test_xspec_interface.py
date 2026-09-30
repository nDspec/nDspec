"""Tests for ndspec.XspecInterface.

Most of these tests do not need HEASOFT. A mock model library is compiled on
the fly from C source with whatever C compiler is available. It exports one
routine per Xspec calling convention, written with the exact ABI Xspec uses:

* single/double precision Fortran (``name_``, every argument by reference),
* the C-style double precision interface (``name`` and ``C_name``),
* ``FNINIT``,

plus a few getters so the tests can check what each routine actually received
(number of bins, spectrum number, init string, call count). A C function with
Fortran-style arguments is ABI-identical to a gfortran subroutine, so no
Fortran compiler is needed; when gfortran is available, one extra test also
checks a real Fortran subroutine.

Tests at the bottom run against a real Xspec library: a HEASOFT installation
(if $HEADAS is set) and/or the xspectrampoline package (if installed and
$HEADAS is not set). Each is skipped when unavailable.
"""
import ctypes as ct
import importlib.util
import os
import types
import re
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pytest

import ndspec.XspecInterface as X

LIBEXT = ".dylib" if sys.platform == "darwin" else ".so"

# Mock model file and library
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

static int fninit_calls = 0, calls = 0, last_ne = -1, last_spec = -1;
static char last_init[64] = "unset";

#define CSIG const double* e, int n, const double* p, int s, \
             double* f, double* fe, const char* init
#define RECORD_C  calls++; last_ne = n; last_spec = s; \
                  strncpy(last_init, init ? init : "(null)", 63);
#define RECORD_F  calls++; last_ne = *n; last_spec = *ifl;

void FNINIT(void) { fninit_calls++; }
int mock_fninit_calls(void) { return fninit_calls; }
int mock_calls(void)        { return calls; }
int mock_last_ne(void)      { return last_ne; }
int mock_last_spec(void)    { return last_spec; }
const char* mock_last_init(void) { return last_init; }

/* Fortran, single precision: flux = Gamma * bin width */
void donthcomp_(const float* e, const int* n, const float* p, const int* ifl,
                float* f, float* fe)
{ RECORD_F for (int i = 0; i < *n; i++) f[i] = p[0] * (e[i+1] - e[i]); }

/* Fortran, double precision: flux = H */
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

/* multiplicative model that only works after FNINIT */
void C_tbabs(CSIG) { RECORD_C for (int i = 0; i < n; i++) f[i] = fninit_calls ? p[0] : -1.0; }

/* C-style (c_ prefix, symbol without prefix): flux = M * bin width */
void beckerwolff(CSIG) { RECORD_C for (int i = 0; i < n; i++) f[i] = p[1] * (e[i+1] - e[i]); }

/* fallbacks: only the C_ wrapper exists for these */
void C_xsblbd(CSIG) { RECORD_C for (int i = 0; i < n; i++) f[i] = 3.0 * (e[i+1] - e[i]); }
void C_cfall(CSIG)  { RECORD_C for (int i = 0; i < n; i++) f[i] = 7.0; }

/* convolution: scales the input spectrum in place */
void C_gsmooth(CSIG) { RECORD_C for (int i = 0; i < n; i++) f[i] *= p[0]; }
"""

LOCAL_SRC = r"""
/* a 'local model package': no FNINIT, one C-style model */
void C_gaussianLine(const double* e, int n, const double* p, int s,
                    double* f, double* fe, const char* init)
{ for (int i = 0; i < n; i++) f[i] = 2.0 * (e[i+1] - e[i]); }
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


def _build(src, path, compiler=None, suffix=".c"):
    compiler = compiler or _c_compiler()
    if compiler is None:
        pytest.skip("no C compiler available to build the mock model library")
    srcfile = str(path) + suffix
    with open(srcfile, "w") as fh:
        fh.write(src)
    subprocess.check_call([compiler, "-shared", "-fPIC", "-O0", "-o", str(path), srcfile])
    return str(path)


def _stub(name, con=False):
    f = (lambda ear, params, seed: None) if con else (lambda ear, params: None)
    f.__name__ = name
    return f


def _write(tmp_path, text, name="model.dat"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


@pytest.fixture(scope="module")
def mock_paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("xsmock")
    # Never give a mock the name of a real HEASOFT library: on macOS, dlopen()
    # looks up the file name in $DYLD_LIBRARY_PATH *before* the path it was
    # given, so with HEASOFT initialised a mock called libXSFunctions.dylib
    # would silently load the real library instead.
    lib = _build(MOCK_SRC, d / ("libndspec_xsmock" + LIBEXT))
    dat = d / "model.dat"
    dat.write_text(MODEL_DAT)
    return lib, str(dat)


@pytest.fixture(scope="module")
def lib(mock_paths):
    return X.CInterface(*mock_paths)


@pytest.fixture
def fresh_lib(mock_paths, tmp_path):
    """A private build of the mock library, so FNINIT/counter state is fresh.

    The library is rebuilt rather than copied: a copied dylib keeps the
    install name of the original, and macOS dlopen() hands back an
    already-loaded image whose install name matches the requested path, so
    a later load of the original would silently return the copy.
    """
    lib = _build(MOCK_SRC, tmp_path / ("libndspec_xsfresh" + LIBEXT))
    return lib, mock_paths[1]


def _getter(interface, name, restype=ct.c_int):
    f = getattr(interface.lib, name)
    f.restype = restype
    f.argtypes = []
    return f()



@pytest.mark.parametrize("func_call, expected", [
    # no prefix: single-precision Fortran, lower-case symbol; C_ wrapper fallback
    ("doNthComp", [("donthcomp_", X.F77_SINGLE), ("C_doNthComp", X.C_DOUBLE)]),
    ("F_ismabs", [("ismabs_", X.F77_DOUBLE), ("C_ismabs", X.C_DOUBLE)]),
    ("c_beckerwolff", [("beckerwolff", X.C_DOUBLE), ("C_beckerwolff", X.C_DOUBLE),
                       ("c_beckerwolff", X.C_DOUBLE)]),
    ("C_gaussianLine", [("C_gaussianLine", X.C_DOUBLE)]),
    # the prefix is removed as a prefix (the old code used .strip('c_'))
    ("c_simpc", [("simpc", X.C_DOUBLE), ("C_simpc", X.C_DOUBLE), ("c_simpc", X.C_DOUBLE)]),
])
def test_resolve_symbols(func_call, expected):
    assert X.resolve_symbols(func_call) == expected


def test_parser(tmp_path):
    """MODEL_DAT has irregular blank lines (none, several, whitespace-only,
    one inside an entry), quoted multi-word units, switch/scale parameters
    and a periodic flag."""
    info = X.parse_model_file(_write(tmp_path, MODEL_DAT))
    assert len(info) == 13
    assert info["nthcomp"]["func_call"] == "donthcomp"      # names lower-cased
    assert info["gaussian"]["func_call"] == "C_gaussianLine"
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


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


@pytest.fixture
def heasoft_tree(tmp_path):
    root = tmp_path / "heasoft-6.36"
    headas = root / "aarch64-apple-darwin25.3.0"
    headas.mkdir(parents=True)
    libname = "libXSFunctions" + (".dylib" if X.platform == "darwin" else ".so")
    return {
        "headas": headas,
        "install": (headas / "lib" / libname, root / "spectral" / "manager" / "model.dat"),
        "source": (root / "Xspec" / headas.name / "lib" / libname,
                   root / "Xspec" / "src" / "manager" / "model.dat"),
    }


@pytest.mark.parametrize("layouts, expected", [
    (["install"], "install"),
    (["source"], "source"),
    (["install", "source"], "install"),
])
def test_find_heasoft(heasoft_tree, monkeypatch, layouts, expected):
    for layout in layouts:
        for f in heasoft_tree[layout]:
            _touch(f)
    # a trailing slash on HEADAS used to break the source-layout path
    monkeypatch.setenv("HEADAS", str(heasoft_tree["headas"]) + "/")
    assert X.find_heasoft() == tuple(os.path.normpath(str(f)) for f in heasoft_tree[expected])


def test_find_heasoft_errors(heasoft_tree, monkeypatch):
    monkeypatch.delenv("HEADAS", raising=False)
    with pytest.raises(EnvironmentError, match="HEADAS"):
        X.find_heasoft()
    lib, dat = heasoft_tree["install"]
    _touch(lib)
    with pytest.raises(FileNotFoundError, match="model.dat"):
        X.find_heasoft(str(heasoft_tree["headas"]))


def test_fninit_once_per_library(fresh_lib):
    a = X.CInterface(*fresh_lib, initialize=False)
    assert _getter(a, "mock_fninit_calls") == 0
    a.add_model(_stub("tbabs"))
    assert np.all(a.tbabs([1.0, 2.0], [0.5]) == -1.0)      # not initialised yet
    X.CInterface(*fresh_lib)                                 # auto-detects FNINIT
    X.FortranInterface(*fresh_lib)
    a.initialize_heasoft()
    assert _getter(a, "mock_fninit_calls") == 1
    assert np.allclose(a.tbabs([1.0, 2.0], [0.5]), 0.5)


def test_library_without_fninit(tmp_path):
    """Local model packages do not export FNINIT."""
    local = _build(LOCAL_SRC, tmp_path / ("liblocal" + LIBEXT))
    dat = _write(tmp_path, "gaussian 2 0. 1e20 C_gaussianLine add 0\n"
                           "LineE keV 6.5 0 0 1e6 1e6 0.05\n"
                           "Sigma keV 0.1 0 0 10 20 0.05\n")
    lib = X.CInterface(local, dat)
    lib.add_model(_stub("gaussian"))
    assert np.allclose(lib.gaussian([1.0, 2.0], [6.5, 0.1, 1.0]), 2.0)
    with pytest.raises(AttributeError, match="FNINIT"):
        X.CInterface(local, dat, initialize=True)


def test_fortran_interface_defaults(mock_paths, monkeypatch):
    """FortranInterface() takes its paths from find_xspec (not loaded from a
    fake $HEADAS tree: the file would have to be called libXSFunctions, which
    macOS resolves through $DYLD_LIBRARY_PATH first)."""
    calls = []
    monkeypatch.setattr(X, "find_xspec", lambda backend: calls.append(backend) or mock_paths)
    lib = X.FortranInterface(backend="xspectrampoline")
    assert (lib.lib_path, lib.pars_path) == mock_paths
    assert calls == ["xspectrampoline"]


@pytest.fixture
def fake_xspectrampoline(monkeypatch):
    """Replace find_heasoft with a recorder and xspectrampoline with a stub."""
    calls = []
    monkeypatch.setattr(X, "find_heasoft",
                        lambda headas=None: calls.append(headas) or ("/x/lib.so", "/x/model.dat"))
    stub = types.ModuleType("xspectrampoline")
    stub.get_HEADAS = lambda: "/bundle/LibXSPEC"
    monkeypatch.setitem(sys.modules, "xspectrampoline", stub)
    return calls


def test_find_xspec_backends(fake_xspectrampoline, monkeypatch):
    calls = fake_xspectrampoline
    monkeypatch.setenv("HEADAS", "/my/heasoft")
    X.find_xspec("auto")                      # HEADAS set -> HEASOFT
    X.find_xspec("heasoft")
    assert calls == [None, None]
    monkeypatch.delenv("HEADAS")
    X.find_xspec("auto")                      # no HEADAS -> xspectrampoline's library
    X.find_xspec("xspectrampoline")
    assert calls[2:] == ["/bundle/LibXSPEC"] * 2
    # FNINIT already ran when xspectrampoline was imported
    assert os.path.realpath("/x/lib.so") in X._INITIALISED_LIBS
    with pytest.raises(ValueError, match="backend"):
        X.find_xspec("pyxspec")


def test_find_xspec_without_xspectrampoline(fake_xspectrampoline, monkeypatch, tmp_path):
    monkeypatch.delenv("HEADAS", raising=False)
    monkeypatch.setitem(sys.modules, "xspectrampoline", None)     # not installed
    with pytest.raises(EnvironmentError, match="pip install xspectrampoline"):
        X.find_xspec("auto")
    with pytest.raises(ImportError, match="not installed"):
        X.find_xspec("xspectrampoline")
    # installed, but its import fails (it raises NoLibXSPEC if it cannot load
    # its libraries, and AttributeError on Python 3.8)
    _touch(tmp_path / "xspectrampoline" / "__init__.py").write_text(
        "raise RuntimeError('NoLibXSPEC')\n")
    monkeypatch.delitem(sys.modules, "xspectrampoline")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ImportError, match="could not load.*NoLibXSPEC"):
        X.find_xspec("xspectrampoline")


def test_dyld_shadowing(tmp_path, monkeypatch):
    wanted = _touch(tmp_path / "mine" / "libfoo.dylib")
    other = _touch(tmp_path / "heasoft" / "lib" / "libfoo.dylib")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", str(other.parent))
    monkeypatch.setattr(X, "platform", "darwin")
    assert X._dyld_shadowing(str(wanted)) == str(other)
    monkeypatch.setattr(X, "platform", "linux")   # Linux has no such lookup
    assert X._dyld_shadowing(str(wanted)) is None

@pytest.mark.parametrize("model, symbol, abi", [
    ("nthcomp", "donthcomp_", X.F77_SINGLE),
    ("ismabs", "ismabs_", X.F77_DOUBLE),
    ("gaussian", "C_gaussianLine", X.C_DOUBLE),
    ("bwcycl", "beckerwolff", X.C_DOUBLE),
    ("blbd_like", "C_xsblbd", X.C_DOUBLE),   # Fortran symbol missing -> C_ fallback
    ("cfall", "C_cfall", X.C_DOUBLE),        # c_ base symbol missing -> C_ fallback
])
def test_binding(lib, model, symbol, abi):
    lib.add_model(_stub(model))
    assert (lib.models_info[model]["symbol"], lib.models_info[model]["abi"]) == (symbol, abi)
    assert "symbol" not in lib._all_info[model]    # the shared table is not mutated


def test_add_model_errors(lib):
    with pytest.raises(KeyError, match="notamodel"):
        lib.add_model(_stub("notamodel"))
    with pytest.raises(NotImplementedError, match="mix"):
        lib.add_model(_stub("mixmod"))
    with pytest.raises(AttributeError, match='(?s)C_relxcpp.*extern "C"'):
        lib.add_model(_stub("relxcpp"))
    assert not hasattr(lib, "relxcpp")
    assert sorted(lib.check_symbols()[1]) == ["mixmod", "relxcpp"]


def test_symbol_and_language_override(lib):
    lib.add_model(_stub("prec4"), symbol="prec8_", language=X.F77_DOUBLE)
    assert lib.prec4([1.0, 2.0], [1.0 / 3.0])[0] == 1.0 / 3.0
    lib.add_model(_stub("blbd_like"), language=X.C_DOUBLE)
    assert lib.models_info["blbd_like"]["symbol"] == "C_xsblbd"
    with pytest.raises(AttributeError):
        lib.add_model(_stub("gaussian"), language=X.F77_DOUBLE)

EAR = np.array([1.0, 1.5, 2.5, 4.0, 8.0])   # non-uniform bins


@pytest.fixture
def loaded(lib):
    for m in ["nthcomp", "gaussian", "tbabs", "ismabs", "bwcycl", "prec4", "prec8", "precc"]:
        lib.add_model(_stub(m))
    lib.add_model(_stub("gsmooth", con=True))
    return lib


@pytest.mark.parametrize("model, params, k", [
    ("nthcomp", [2.0, 100, 0.1, 0, 0], 2.0),    # f77 single
    ("gaussian", [6.5, 0.1], 6.5),              # C_ wrapper
    ("bwcycl", [10, 1.4], 1.4),                 # c_ style
])
def test_additive(loaded, model, params, k):
    """Mocks return flux = k * bin width per bin; ndspec returns flux per keV
    times the norm, which it applies itself."""
    for norm in (0.0, 2.5):
        out = getattr(loaded, model)(EAR, params + [norm])
        assert out.shape == (EAR.size - 1,) and out.dtype == np.float64
        assert np.allclose(out, k * norm)


def test_multiplicative(loaded):
    assert np.allclose(loaded.tbabs(EAR, [0.25]), 0.25)   # no width division, no norm
    assert np.allclose(loaded.ismabs(EAR, [0.3]), 0.3)


def test_convolution(loaded):
    seed = np.array([1.0, 2.0, 3.0, 4.0])
    out = loaded.gsmooth(EAR, [2.0, 0.0], seed)
    assert np.allclose(out, [2.0, 4.0, 6.0, 8.0])
    assert np.array_equal(seed, [1.0, 2.0, 3.0, 4.0])   # input untouched
    with pytest.raises(ValueError, match="seed"):
        loaded.gsmooth(EAR, [2.0, 0.0], np.ones(EAR.size))


@pytest.mark.parametrize("model, dtype", [("prec4", np.float32),
                                          ("prec8", np.float64),
                                          ("precc", np.float64)])
def test_precision(loaded, model, dtype):
    third = 1.0 / 3.0
    assert np.all(getattr(loaded, model)(EAR, [third]) == np.float64(dtype(third)))


@pytest.mark.parametrize("model, params", [("gaussian", [6.5, 0.1, 1.0]),        # C ABI
                                           ("nthcomp", [2.0, 100, 0.1, 0, 0, 1.0])])  # f77
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
    assert np.allclose(loaded.gaussian(big[::2], [6.5, 0.1, 1.0]), ref)       # C
    assert np.allclose(loaded.nthcomp(big[::2], [2.0, 100, 0.1, 0, 0, 1.0]), 2.0)  # Fortran
    # a single edge gives no bins
    with pytest.raises(ValueError, match="ear"):
        loaded.gaussian([1.0], [6.5, 0.1, 1.0])


@pytest.mark.parametrize("params, bad", [
    ([0.5, 100, 0.1, 0, 0, 1.0], "Gamma"),
    ([2.0, 100, 0.1, 0, 20.0, 1.0], "Redshift"),
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


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not available")
def test_real_fortran_subroutines(tmp_path):
    path = tmp_path / ("libndspec_f" + LIBEXT)
    _build(F90_SRC, path, compiler=shutil.which("gfortran"), suffix=".f90")
    dat = _write(tmp_path, "nthComp 1 0. 1e20 donthcomp add 0\n"
                           "Gamma \" \" 1.7 1.001 1.005 5. 10. 0.01\n"
                           "ismabs 1 0. 1e20 F_ismabs mul 0\n"
                           "H 10^22 0.1 0 0 1e5 1e6 1e-3\n")
    lib = X.CInterface(str(path), dat)
    lib.load_models({"nthcomp": _stub("nthcomp"), "ismabs": _stub("ismabs")})
    # the subroutines add (ifl - 1) and ne, so both must arrive by reference intact
    assert np.allclose(lib.nthcomp(EAR, [2.0, 1.0]), 2.0)
    assert np.allclose(lib.ismabs(EAR, [0.25]), 0.25 + (EAR.size - 1))


_USER_HEADAS = os.environ.get("HEADAS")
_HAVE_XSPECTRAMPOLINE = (sys.version_info >= (3, 9)
                         and importlib.util.find_spec("xspectrampoline") is not None)


@pytest.fixture(scope="module", params=["heasoft", "xspectrampoline"])
def real_lib(request):
    if request.param == "heasoft":
        if not _USER_HEADAS:
            pytest.skip("HEADAS not set: no HEASOFT installation to test")
        return X.FortranInterface(*X.find_heasoft(_USER_HEADAS))
    if not _HAVE_XSPECTRAMPOLINE:
        pytest.skip("xspectrampoline not installed")
    if _USER_HEADAS:
        # xspectrampoline would use that HEASOFT instead of its bundled copy
        pytest.skip("HEADAS is set, so xspectrampoline would not use its own library")
    return X.FortranInterface(backend="xspectrampoline")


_HEADER = re.compile(r"^(\S+)\s+(\d+)\s+\S+\s+\S+\s+(\S+)\s+(add|mul|con|mix|acn|amx)\b")


def test_real_model_dat_and_symbols(real_lib):
    """Every model in the real model.dat parses with the declared number of
    parameters and resolves to a symbol in the real library."""
    expected = {}
    with open(real_lib.pars_path) as fh:
        for line in fh:
            m = _HEADER.match(line)
            if m:
                expected[m.group(1).lower()] = int(m.group(2)) + (m.group(4) == "add")
    assert {k: len(v["parameters"]) for k, v in real_lib._all_info.items()} == expected
    assert real_lib.check_symbols()[1] == []


def test_real_models_evaluate(real_lib):
    # Only models that need no data files: xspectrampoline does not ship
    # spectral/modelData, and some models (e.g. kerrbb) exit the process
    # when a data file is missing.
    for model in ["powerlaw", "tbabs", "nthcomp", "gaussian", "diskbb"]:
        real_lib.add_model(_stub(model))
        pars = [v["value"] for v in real_lib._all_info[model]["parameters"].values()]
        out = getattr(real_lib, model)(np.logspace(-1, 2, 200), pars)
        assert np.all(np.isfinite(out)) and np.any(out != 0), model
