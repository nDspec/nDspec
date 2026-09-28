"""Tests for ndspec.XspecInterface that do not need a HEASOFT installation.

A small mock library exporting one routine per Xspec calling convention is
compiled on the fly; those tests are skipped if no Fortran/C++ compiler is
available.
"""
import os
import shutil
import subprocess
import warnings

import numpy as np
import pytest

import ndspec.XspecInterface as X

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
gsmooth   2  0. 1.e20  C_gsmooth  con 0
Sig_6keV keV 1.0 0 0 10 20 0.05
Index " " 0 -1 -1 1 1 -0.01 P
"""

F77_SRC = """\
subroutine donthcomp(ear, ne, param, ifl, photar, photer)
  integer ne, ifl
  real*4 ear(0:ne), param(*), photar(ne), photer(ne)
  integer i
  do i = 1, ne
     photar(i) = param(1) * (ear(i) - ear(i-1))
  end do
end subroutine
subroutine ismabs(ear, ne, param, ifl, photar, photer)
  integer ne, ifl
  real*8 ear(0:ne), param(*), photar(ne), photer(ne)
  integer i
  do i = 1, ne
     photar(i) = param(1)
  end do
end subroutine
"""

CXX_SRC = r"""
static int inited = 0;
#define SIG const double* e, int n, const double* p, int s, double* f, double* fe, const char* init
extern "C" void FNINIT(void) { inited = 1; }
extern "C" void C_gaussianLine(SIG) { for (int i=0;i<n;i++) f[i] = p[0]*(e[i+1]-e[i]); }
extern "C" void C_tbabs(SIG)        { for (int i=0;i<n;i++) f[i] = inited ? p[0] : -1.0; }
extern "C" void beckerwolff(SIG)    { for (int i=0;i<n;i++) f[i] = p[1]*(e[i+1]-e[i]); }
extern "C" void C_xsblbd(SIG)       { for (int i=0;i<n;i++) f[i] = 3.0*(e[i+1]-e[i]); }
extern "C" void C_gsmooth(SIG)      { for (int i=0;i<n;i++) f[i] *= p[0]; }
"""


def _stub(name, con=False):
    f = (lambda ear, params, seed: None) if con else (lambda ear, params: None)
    f.__name__ = name
    return f


@pytest.fixture(scope="module")
def model_dat(tmp_path_factory):
    path = tmp_path_factory.mktemp("xs") / "model.dat"
    path.write_text(MODEL_DAT)
    return str(path)


@pytest.fixture(scope="module")
def mock_lib(tmp_path_factory, model_dat):
    fc, cxx = shutil.which("gfortran"), shutil.which("g++") or shutil.which("clang++")
    if not (fc and cxx):
        pytest.skip("gfortran and a C++ compiler are needed for the mock library")
    d = tmp_path_factory.mktemp("xslib")
    (d / "f.f90").write_text(F77_SRC)
    (d / "c.cxx").write_text(CXX_SRC)
    ext = ".dylib" if X.platform == "darwin" else ".so"
    lib = d / ("libXSFunctions" + ext)
    subprocess.check_call([fc, "-fPIC", "-c", "f.f90", "-o", "f.o"], cwd=d)
    subprocess.check_call([cxx, "-fPIC", "-c", "c.cxx", "-o", "c.o"], cwd=d)
    subprocess.check_call([fc, "-shared", "-o", str(lib), "f.o", "c.o",
                           "-lc++" if X.platform == "darwin" else "-lstdc++"], cwd=d)
    return X.CInterface(str(lib), model_dat)


def test_resolve_symbols():
    assert X.resolve_symbols("donthcomp")[0] == ("donthcomp_", X.F77_SINGLE)
    assert X.resolve_symbols("F_ismabs")[0] == ("ismabs_", X.F77_DOUBLE)
    assert X.resolve_symbols("c_beckerwolff")[0] == ("beckerwolff", X.C_DOUBLE)
    assert X.resolve_symbols("C_gaussianLine") == [("C_gaussianLine", X.C_DOUBLE)]
    # the prefix is stripped as a prefix, not as a character set
    assert X.resolve_symbols("c_simpc")[0] == ("simpc", X.C_DOUBLE)


def test_parser(model_dat):
    info = X.parse_model_file(model_dat)
    assert set(info) == {"nthcomp", "gaussian", "tbabs", "ismabs", "bwcycl",
                         "blbd_like", "gsmooth"}
    assert list(info["nthcomp"]["parameters"]) == [
        "Gamma", "kT_e", "kT_bb", "inp_type", "Redshift", "norm"]
    assert info["nthcomp"]["func_call"] == "donthcomp"
    assert info["tbabs"]["parameters"]["nH"]["unit"] == "10^22 atoms cm^-2"
    assert info["gsmooth"]["parameters"]["Index"]["max"] == 1.0


def test_every_convention(mock_lib):
    for n in ["nthcomp", "gaussian", "tbabs", "ismabs", "bwcycl", "blbd_like"]:
        mock_lib.add_model(_stub(n))
    mock_lib.add_model(_stub("gsmooth", con=True))
    ear = np.linspace(1, 10, 11)
    assert np.allclose(mock_lib.nthcomp(ear, [2.0, 100, 0.1, 0, 0, 5.0]), 10.0)
    assert np.allclose(mock_lib.gaussian(ear, [6.5, 0.1, 2.0]), 13.0)
    assert np.allclose(mock_lib.tbabs(ear, [0.7]), 0.7)          # FNINIT was run
    assert np.allclose(mock_lib.ismabs(ear, [0.3]), 0.3)
    assert np.allclose(mock_lib.bwcycl(ear, [10, 1.4, 1.0]), 1.4)
    assert np.allclose(mock_lib.blbd_like(ear, [3.0, 1.0]), 3.0)  # C_ fallback
    seed = np.ones(10)
    assert np.allclose(mock_lib.gsmooth(ear, [2.0, 0.0], seed), 2.0)
    assert np.all(seed == 1.0)
    resolved, missing = mock_lib.check_symbols()
    assert missing == []


def test_bounds_checked_for_all_parameters(mock_lib):
    mock_lib.add_model(_stub("nthcomp"))
    ear = np.linspace(1, 10, 11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = mock_lib.nthcomp(ear, [2.0, 100, 0.1, 0, 20.0, 5.0])  # Redshift > 10
    assert np.all(np.isnan(out))
