import ctypes as ct
import os
import shlex
import warnings
from functools import wraps
from sys import platform

import numpy as np

__all__ = ["ModelInterface", "FortranInterface", "CInterface",
           "find_heasoft", "find_xspectrampoline", "find_xspec",
           "resolve_symbols"]

# Calling conventions
F77_SINGLE = "f77_single"   # subroutine f(ear,ne,par,ifl,photar,photer), real*4
F77_DOUBLE = "f77_double"   # same, real*8
C_DOUBLE = "c_double"       # void f(double*,int,double*,int,double*,double*,char*)
_ABIS = (F77_SINGLE, F77_DOUBLE, C_DOUBLE)

_DTYPES = {F77_SINGLE: np.float32, F77_DOUBLE: np.float64, C_DOUBLE: np.float64}
_CTYPES = {F77_SINGLE: ct.c_float, F77_DOUBLE: ct.c_double, C_DOUBLE: ct.c_double}

_SUPPORTED_TYPES = ("add", "mul", "con")

# FNINIT must run once per loaded library, not once per interface object.
_INITIALISED_LIBS = set()

def find_heasoft(headas=None):
    """
    Locate the Xspec model library and model.dat of a HEASOFT installation.

    Both the documented install layout ($HEADAS/lib and
    $HEADAS/../spectral/manager, used by binary and conda installs) and the
    source-build layout ($HEADAS/../Xspec/<platform>/lib and
    $HEADAS/../Xspec/src/manager) are searched, in that order.

    Parameters:
    -----------
    headas: str, optional
        Path to the HEASOFT platform directory. Defaults to $HEADAS.

    Output:
    -------
    lib_path, pars_path: str, str
        Paths to libXSFunctions.{so,dylib} and model.dat.
    """
    headas = headas or os.environ.get("HEADAS")
    if not headas:
        raise EnvironmentError(
            "HEADAS environment variable not set. Initialise HEASOFT "
            "(source $HEADAS/headas-init.sh) or pass lib_path and pars_path.")
    # normpath also removes a trailing '/', which would otherwise make
    # basename() return an empty string
    headas = os.path.normpath(os.path.expanduser(headas))

    if platform.startswith("linux"):
        libname = "libXSFunctions.so"
    elif platform == "darwin":
        libname = "libXSFunctions.dylib"
    else:
        raise OSError(f"Platform {platform} is not supported.")

    lib_candidates = [
        os.path.join(headas, "lib", libname),
        os.path.join(headas, "..", "Xspec", os.path.basename(headas), "lib", libname),
    ]
    dat_candidates = [
        os.path.join(headas, "..", "spectral", "manager", "model.dat"),
        os.path.join(headas, "..", "Xspec", "src", "manager", "model.dat"),
        os.path.join(headas, "spectral", "manager", "model.dat"),
    ]

    lib_path = next((p for p in lib_candidates if os.path.exists(p)), None)
    pars_path = next((p for p in dat_candidates if os.path.exists(p)), None)
    if lib_path is None:
        raise FileNotFoundError(
            f"Could not find {libname}; looked in:\n  " + "\n  ".join(lib_candidates))
    if pars_path is None:
        raise FileNotFoundError(
            "Could not find model.dat; looked in:\n  " + "\n  ".join(dat_candidates))
    return os.path.normpath(lib_path), os.path.normpath(pars_path)


def _dyld_shadowing(lib_path):
    """
    On macOS, dlopen() first looks for the *file name* of lib_path in every
    $DYLD_LIBRARY_PATH directory, and only then at lib_path itself. Return
    the library that would be loaded instead of lib_path, or None.
    """
    if platform != "darwin":
        return None
    leaf = os.path.basename(lib_path)
    for d in os.environ.get("DYLD_LIBRARY_PATH", "").split(os.pathsep):
        candidate = os.path.join(d, leaf) if d else ""
        if candidate and os.path.exists(candidate):
            try:
                if os.path.samefile(candidate, lib_path):
                    return None
            except OSError:
                pass
            return candidate
    return None


def find_xspectrampoline():
    """
    Locate the Xspec model library bundled by xspectrampoline
    (``pip install xspectrampoline``), which needs no HEASOFT installation.

    Importing xspectrampoline loads the library and runs FNINIT. If $HEADAS is
    already set, xspectrampoline (and therefore this function) uses that
    HEASOFT installation instead of its bundled copy.

    The bundle does not include the model data directory
    (spectral/modelData), so models that read data files (e.g. kerrbb, apec)
    fail until the files are copied into
    ``xspectrampoline_helpers.get_model_data_dir()``. Some of them stop the
    Python process outright when a file is missing.

    Output:
    -------
    lib_path, pars_path: str, str
        Paths to libXSFunctions.{so,dylib} and model.dat.
    """
    try:
        import xspectrampoline
    except ImportError as exc:
        raise ImportError(
            "xspectrampoline is not installed. Install it with "
            "'pip install xspectrampoline' (Python >= 3.9, Linux x86_64 or "
            "macOS), or set HEADAS to use a HEASOFT installation.") from exc
    except Exception as exc:    # NoLibXSPEC, or an unsupported Python version
        raise ImportError(
            f"xspectrampoline is installed but could not load its Xspec "
            f"library: {exc!r}") from exc
    lib_path, pars_path = find_heasoft(xspectrampoline.get_HEADAS())
    # importing xspectrampoline has already run FNINIT on this library
    _INITIALISED_LIBS.add(os.path.realpath(lib_path))
    return lib_path, pars_path


_BACKENDS = ("auto", "heasoft", "xspectrampoline")


def find_xspec(backend="auto"):
    """
    Locate an Xspec model library and its model.dat.

    Parameters:
    -----------
    backend: str
        "heasoft": a HEASOFT installation found via $HEADAS.
        "xspectrampoline": the library bundled by the xspectrampoline package.
        "auto" (default): HEASOFT if $HEADAS is set, otherwise xspectrampoline.

    Output:
    -------
    lib_path, pars_path: str, str
    """
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, not {backend!r}")
    if backend == "heasoft":
        return find_heasoft()
    if backend == "xspectrampoline":
        return find_xspectrampoline()
    if os.environ.get("HEADAS"):
        return find_heasoft()
    try:
        return find_xspectrampoline()
    except ImportError as exc:
        raise EnvironmentError(
            "No Xspec model library found. Either set HEADAS to a HEASOFT "
            "installation (source $HEADAS/headas-init.sh), or run "
            f"'pip install xspectrampoline'.\n({exc})") from exc


def resolve_symbols(func_call):
    """
    Return the ordered list of (symbol, abi) candidates for a model.dat
    function field. The first entry is the one Xspec itself would use; the
    others are ABI-safe fallbacks. The ABI of every candidate is fixed by the
    form of the symbol, so a symbol is never called with the wrong signature.

    Parameters:
    -----------
    func_call: str
        The function field of the model.dat entry, e.g. "C_gaussianLine".

    Output:
    -------
    candidates: list of (str, str)
        (symbol name, calling convention) pairs.
    """
    if func_call.startswith("C_"):
        base = func_call[2:]
        cands = [("C_" + base, C_DOUBLE)]
    elif func_call.startswith("c_"):
        base = func_call[2:]
        cands = [(base, C_DOUBLE), ("C_" + base, C_DOUBLE), ("c_" + base, C_DOUBLE)]
    elif func_call.startswith("F_"):
        base = func_call[2:]
        cands = [(base.lower() + "_", F77_DOUBLE), ("C_" + base, C_DOUBLE)]
    else:
        base = func_call
        cands = [(base.lower() + "_", F77_SINGLE), ("C_" + base, C_DOUBLE)]
    # de-duplicate while preserving order
    seen, out = set(), []
    for c in cands:
        if c[0] not in seen:
            seen.add(c[0])
            out.append(c)
    return out


def _abi_for_user_symbol(symbol, func_call):
    """Infer the calling convention of a user-supplied symbol from its form."""
    if symbol.endswith("_") and not symbol.startswith("C_"):
        return F77_DOUBLE if func_call.startswith("F_") else F77_SINGLE
    return C_DOUBLE

def _tokens(line):
    try:
        return shlex.split(line, posix=True)
    except ValueError:          # unbalanced quotes, apostrophes in units, ...
        return line.split()


def _is_float(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _parse_parameter(line):
    parts = _tokens(line)
    if not parts:
        return None
    name = parts[0]
    if name[0] in "$*":
        # switch ($) or scale (*) parameter: a single value, no fit bounds
        nums = [float(t) for t in parts[1:] if _is_float(t)]
        return name.strip("$*"), {"value": nums[0] if nums else 0.0,
                                  "min": None, "max": None, "unit": "n/a"}
    # regular: name unit value hardmin softmin softmax hardmax delta [P]
    rest = parts[1:]
    if rest and rest[-1].upper() == "P":            # periodic flag
        rest = rest[:-1]
    ntail = 0
    for tok in reversed(rest):
        if not _is_float(tok):
            break
        ntail += 1
    if ntail < 6:
        raise ValueError(f"cannot parse parameter line: {line!r}")
    if ntail > 6 and len(rest) == ntail:            # unit omitted entirely
        ntail = len(rest)
    nums = [float(t) for t in rest[-6:]]
    unit = " ".join(rest[:len(rest) - 6]).strip() or "n/a"
    return name, {"value": nums[0], "min": nums[1], "max": nums[4], "unit": unit}


def parse_model_file(input_file):
    """
    Parse an Xspec model.dat/lmodel.dat file. Entries are read the same way
    Xspec reads them: a header line followed by exactly <npars> parameter
    lines, so irregular blank lines or whitespace do not merge models.
    """
    with open(input_file, "r") as fh:
        lines = [ln.rstrip("\n") for ln in fh]

    models_info = {}
    i, n = 0, len(lines)
    while i < n:
        parts = lines[i].split()
        i += 1
        if len(parts) < 6 or parts[0].startswith("#"):
            continue
        try:
            npars = int(parts[1])
        except ValueError:
            continue
        model_name, func_call, model_type = parts[0].lower(), parts[4], parts[5]

        parameters = {}
        read = 0
        while read < npars and i < n:
            line = lines[i]
            i += 1
            if not line.strip():
                continue
            read += 1
            try:
                parsed = _parse_parameter(line)
            except (ValueError, IndexError) as exc:
                warnings.warn(f"{model_name}: {exc}", UserWarning)
                continue
            if parsed is not None:
                parameters[parsed[0]] = parsed[1]

        if model_type == "add":
            parameters["norm"] = {"value": 1.0, "min": 0.0, "max": 1e20, "unit": "n/a"}
        models_info[model_name] = {"func_call": func_call,
                                   "type": model_type,
                                   "parameters": parameters}
    return models_info


class ModelInterface():
    """
    This class allows users to load a library file containing Xspec-compatible
    models (including the entire library that comes with a typical HEASOFT
    installation), initialize them into the class objects as class methods,
    and evaluate them in their own Python code. The symbol and calling
    convention of each model are derived automatically from the function field
    of its model.dat entry, so Fortran, C and C++ models can be mixed freely.

    Attributes:
    -----------
    models_info: dict
        A dictionary to store model information. The keywords store the
        initialized model names (models_info['nthcomp']), the model type and
        parameters (e.g. models_info[['nthcomp']['type']), the names,
        minimum and maximum parameter values, and units of each (e.g.
        models_info[['nthcomp']['parameters']['kTe']['unit'] = "keV"), and the
        library symbol and calling convention used ('symbol', 'abi').

    lib:  DLL ctype
        The compiled .so (Linux) or .dylib (MacOS) library file loaded.

    _all_info: dict
        A dictionary containing the information of every model in the loaded
        library, regardless of whether the user has initialized it for use or
        not. The structure is identical to models_info.
    """
    def __init__(self, lib_path, pars_path, initialize=None):
        self._all_info = self.parse_models(pars_path)
        self.models_info = {}
        self.lib_path = lib_path
        self.pars_path = pars_path
        # RTLD_GLOBAL lets libraries loaded later (e.g. local model packages)
        # resolve the XSPEC utility symbols exported by this one.
        self.lib = ct.CDLL(lib_path, mode=ct.RTLD_GLOBAL)
        shadow = _dyld_shadowing(lib_path)
        if shadow is not None:
            warnings.warn(
                f"{shadow} is loaded instead of {lib_path}: on macOS, "
                "DYLD_LIBRARY_PATH is searched for the library's file name "
                "before the path given. Rename the library or remove that "
                "directory from DYLD_LIBRARY_PATH.", UserWarning)
        if initialize is None:
            initialize = self._has_symbol("FNINIT") or self._has_symbol("fninit_")
        if initialize:
            self.initialize_heasoft()

    def parse_models(self, input_file):
        """
        This method parses the Xspec file with the model name, type, parameters
        values and units, etc, and stores the necessary information in the
        _all_info dictionary.

        Parameters:
        -----------
        input_file: str
            A path to the model.dat/lmodel.dat file to be parsed.

        Output:
        -------
        models_info: dict
            A dictionary containing model names and types, parameter values,
            minimum and maximum bounds, and parameter units for all models in
            the library.
        """
        return parse_model_file(input_file)

    def _has_symbol(self, name):
        try:
            getattr(self.lib, name)
            return True
        except AttributeError:
            return False

    def initialize_heasoft(self):
        """
        This method calls the FNINIT HEASOFT function, which initializes cross
        sections, abundances and model data paths, and is required to correctly
        evaluate Xspec models outside of the Xspec command interface. It is run
        only once per library.
        """
        key = os.path.realpath(self.lib_path)
        if key in _INITIALISED_LIBS:
            return
        for name in ("FNINIT", "fninit_"):
            if self._has_symbol(name):
                init_call = getattr(self.lib, name)
                init_call.argtypes = []
                init_call.restype = None
                init_call()
                _INITIALISED_LIBS.add(key)
                return
        raise AttributeError(f"{self.lib_path} does not export FNINIT.")

    def print_model_info(self):
        """
        This method prints to terminal a list of all the models that are
        currently initialized and ready for use, as well as their model type,
        library symbol, parameter names, default/min/max values and units.
        """
        print()
        print("Initialized Xspec models:")
        for component, details in self.models_info.items():
            print(f"{component}:")
            print(f"  type: {details['type']}")
            print(f"  function called: {details['func_call']}"
                  f" -> {details.get('symbol')} ({details.get('abi')})")
            print("  parameters:")
            for param, values in details['parameters'].items():
                value_str = ', '.join(f"{key}: {value}" for key, value in values.items())
                print(f"    {param}: {value_str}")
            print()

    def check_param_values(self, model_name, params):
        """
        This method ensures that the input parameters for a given model
        evaluation do not exceed its allowed bounds, by comparing the input
        values with the minimum and maximum stored in the models_info dictionary

        Parameters:
        -----------
        model_name: str
            A string containing the name of the model being computed

        params: np.array
            An array of parameter values being used in the model computation

        Output:
        -------
        test_pars: bool
            False if the number of parameters is wrong or any value is out of
            bounds (in which case the model evaluation returns NaN), True
            otherwise.
        """
        par_data = self.models_info[model_name]['parameters']
        if len(par_data) != len(params):
            warnings.warn(f"Wrong parameter number {len(par_data)} required but "
                          f"{len(params)} passed", UserWarning)
            return False
        for value, (key, info) in zip(params, par_data.items()):
            lo, hi = info['min'], info['max']
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                warnings.warn(f"Model parameter {key} value {value} out of bounds, "
                              f"{info}", UserWarning)
                return False
        return True

    def load_models(self, models):
        """
        This method allows users to initialized multiple models simultaneously
        by passing a dictionary with model names and calling functions.

        Parameters:
        -----------
        models: dict
            A dictionary whose keys are identical to the function names users
            want to intialize in the class.
        """
        for model_name, model_func in models.items():
            self.add_model(model_func)

    def check_symbols(self):
        """
        Report, without calling anything, how every model in the parsed
        model file would be bound in the loaded library.

        Output:
        -------
        resolved, missing: dict, list
            resolved maps model name -> (symbol, abi); missing lists the models
            for which none of the candidate symbols is exported.
        """
        resolved, missing = {}, []
        for name, info in self._all_info.items():
            for sym, abi in resolve_symbols(info['func_call']):
                if self._has_symbol(sym):
                    resolved[name] = (sym, abi)
                    break
            else:
                missing.append(name)
        return resolved, missing

    def _bind(self, func_name, symbol=None, abi=None):
        """Find the library routine for a model and fix its ctypes signature."""
        func_call = self.models_info[func_name]['func_call']
        if symbol is not None:
            candidates = [(symbol, abi or _abi_for_user_symbol(symbol, func_call))]
        else:
            candidates = resolve_symbols(func_call)
            if abi is not None:
                candidates = [c for c in candidates if c[1] == abi]

        for sym, sym_abi in candidates:
            if not self._has_symbol(sym):
                continue
            lib_func = getattr(self.lib, sym)
            real = ct.POINTER(_CTYPES[sym_abi])
            if sym_abi == C_DOUBLE:
                lib_func.argtypes = [real, ct.c_int, real, ct.c_int, real, real,
                                     ct.c_char_p]
            else:   # Fortran passes every argument by reference
                lib_func.argtypes = [real, ct.POINTER(ct.c_int), real,
                                     ct.POINTER(ct.c_int), real, real]
            lib_func.restype = None
            return lib_func, sym, sym_abi

        tried = ", ".join(f"{s} ({a})" for s, a in candidates)
        hint = ""
        if func_call.startswith("C_"):
            hint = ("\nThis is a C++-style model: the library must export an "
                    f"extern \"C\" wrapper named {func_call} (HEASOFT's funcWrappers "
                    "provide these for built-in models; local packages may not).")
        raise AttributeError(
            f"Model '{func_name}' (function '{func_call}' in {self.pars_path}) "
            f"not found in {self.lib_path}. Tried: {tried}.{hint}\n"
            f"Check with: nm -D {self.lib_path} | grep -i {func_call.split('_', 1)[-1]}")

    def add_model(self, func, symbol=None, language=None):
        """
        This method initializes a given model by adding it to the library object
        as one of its methods - for example:

        def powerlaw(ear, params):
            pass

        lib.add_model(powerlaw)
        model = lib.powerlaw(arguments)

        The library symbol and calling convention are derived from the model's
        function field in model.dat, following the same rules Xspec uses.

        Parameters:
        -----------
        func: function
            An empty function with the same name and input parameters as the
            model to be added to the library object.

        symbol: str, optional
            Override the library symbol to call. Only needed for libraries that
            do not follow the Xspec naming conventions. Its calling convention
            is inferred from its form (trailing underscore -> Fortran,
            otherwise C) unless `language` is given.

        language: str, optional
            Force the calling convention: "f77_single", "f77_double" or
            "c_double".
        """
        func_name = func.__name__.rstrip('_').lower()
        if func_name not in self._all_info:
            raise KeyError(f"Model '{func_name}' is not defined in {self.pars_path}")
        if language is not None and language not in _ABIS:
            raise ValueError(f"language must be one of {_ABIS}")

        info = dict(self._all_info[func_name])
        model_type = info['type']
        if model_type not in _SUPPORTED_TYPES:
            raise NotImplementedError(
                f"Model type '{model_type}' ({func_name}) is not supported.")
        self.models_info[func_name] = info

        lib_func, sym, abi = self._bind(func_name, symbol, language)
        info['symbol'], info['abi'] = sym, abi
        dtype = _DTYPES[abi]
        cptr = ct.POINTER(_CTYPES[abi])
        init_string = b""

        def call(ear, params, flux):
            ne = len(ear) - 1
            err = np.zeros(ne, dtype=dtype)
            if abi == C_DOUBLE:
                lib_func(ear.ctypes.data_as(cptr), ct.c_int(ne),
                         params.ctypes.data_as(cptr), ct.c_int(1),
                         flux.ctypes.data_as(cptr), err.ctypes.data_as(cptr),
                         init_string)
            else:
                lib_func(ear.ctypes.data_as(cptr), ct.byref(ct.c_int(ne)),
                         params.ctypes.data_as(cptr), ct.byref(ct.c_int(1)),
                         flux.ctypes.data_as(cptr), err.ctypes.data_as(cptr))
            return flux

        def prepare(ear, params):
            ear = np.ascontiguousarray(ear, dtype=dtype)
            params = np.ascontiguousarray(params, dtype=dtype)
            if ear.ndim != 1 or ear.size < 2:
                raise ValueError("ear must be a 1D array of at least 2 bin edges")
            return ear, params

        if model_type == "add":
            @wraps(func)
            def wrapper(ear, params):
                # check bounds before any cast to float32, so that values at
                # a limit are not pushed across it by rounding
                in_bounds = self.check_param_values(
                    func_name, np.asarray(params, dtype=np.float64))
                ear, params = prepare(ear, params)
                if not in_bounds:
                    return np.full(len(ear) - 1, np.nan)
                # the normalisation is applied here, not passed to the model
                flux = call(ear, np.ascontiguousarray(params[:-1]),
                            np.zeros(len(ear) - 1, dtype=dtype))
                return (flux / np.diff(ear)).astype(np.float64) * float(params[-1])
        elif model_type == "mul":
            @wraps(func)
            def wrapper(ear, params):
                # check bounds before any cast to float32, so that values at
                # a limit are not pushed across it by rounding
                in_bounds = self.check_param_values(
                    func_name, np.asarray(params, dtype=np.float64))
                ear, params = prepare(ear, params)
                if not in_bounds:
                    return np.full(len(ear) - 1, np.nan)
                flux = call(ear, params, np.zeros(len(ear) - 1, dtype=dtype))
                return flux.astype(np.float64)
        else:   # con: the input spectrum is modified in place
            @wraps(func)
            def wrapper(ear, params, seed):
                # check bounds before any cast to float32, so that values at
                # a limit are not pushed across it by rounding
                in_bounds = self.check_param_values(
                    func_name, np.asarray(params, dtype=np.float64))
                ear, params = prepare(ear, params)
                if not in_bounds:
                    return np.full(len(ear) - 1, np.nan)
                flux = np.array(seed, dtype=dtype, copy=True)
                if flux.shape != (len(ear) - 1,):
                    raise ValueError("seed must have len(ear)-1 elements")
                return call(ear, params, flux).astype(np.float64)

        setattr(self, func_name, wrapper)
        return func


class FortranInterface(ModelInterface):
    """
    Interface to the Xspec model library (libXSFunctions). With no path
    arguments the library and model.dat are located according to `backend`
    (see find_xspec), and FNINIT is called:

        FortranInterface()                           # HEASOFT if $HEADAS is set,
                                                     # else xspectrampoline
        FortranInterface(backend="heasoft")          # requires HEASOFT
        FortranInterface(backend="xspectrampoline")  # pip install xspectrampoline

    Despite the (historical) name, Fortran, C and C++ models are all
    supported: the correct symbol and calling convention for each are derived
    from model.dat.
    """
    def __init__(self, lib_path=None, pars_path=None, initialize=None, backend="auto"):
        if lib_path is None or pars_path is None:
            default_lib, default_pars = find_xspec(backend)
            lib_path = lib_path or default_lib
            pars_path = pars_path or default_pars
        ModelInterface.__init__(self, lib_path, pars_path, initialize)


class CInterface(ModelInterface):
    """
    Interface to any other Xspec-compatible model library (e.g. Relxill),
    together with its lmodel.dat file. The calling convention of each model is
    derived from lmodel.dat exactly as for the HEASOFT library, so C, C++ and
    Fortran local models are all supported.
    """
    def __init__(self, lib_path, pars_path, initialize=None):
        ModelInterface.__init__(self, lib_path, pars_path, initialize)
