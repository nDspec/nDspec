import ctypes as ct
import os
import shlex
import warnings

import numpy as np
import xspectrampoline

__all__ = ["ModelInterface", "FortranInterface", "CInterface"]

_XS = xspectrampoline.get_libraries()

_SUPPORTED_TYPES = ("add", "mul", "con")

def _find_model_dat():
    """model.dat of the Xspec installation xspectrampoline loaded."""
    headas = os.path.normpath(str(xspectrampoline.get_HEADAS()))
    candidates = [
        os.path.join(headas, "spectral", "manager", "model.dat"),            # xspectrampoline
        os.path.join(headas, "..", "spectral", "manager", "model.dat"),      # HEASOFT install
        os.path.join(headas, "..", "Xspec", "src", "manager", "model.dat"),  # source build
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.normpath(path)
    raise FileNotFoundError("Could not find model.dat; looked in:\n  " + "\n  ".join(candidates))


def _parse_parameter(line):
    parts = shlex.split(line)
    name = parts[0]
    if name[0] in "$*":
        # switch ($) or scale (*) parameter: "name value" or "name unit value ...",
        # with no fit bounds
        value = float(parts[1] if len(parts) == 2 else parts[2])
        return name.strip("$*"), {"value": value, "min": None, "max": None, "unit": "n/a"}
    # regular: name unit value hardmin softmin softmax hardmax delta [P]
    rest = parts[1:-1] if parts[-1].upper() == "P" else parts[1:]   # periodic flag
    nums = [float(t) for t in rest[-6:]]
    unit = " ".join(rest[:-6]).strip() or "n/a"
    return name, {"value": nums[0], "min": nums[1], "max": nums[4], "unit": unit}


def parse_model_file(input_file):
    """
    Parse an Xspec model.dat/lmodel.dat file. Entries are read the same way
    Xspec reads them: a header line followed by exactly <npars> parameter
    lines, so irregular blank lines or whitespace do not merge models.

    Output:
    -------
    models_info: dict
        {model name: {"func_call", "type", "parameters"}}; "parameters" maps
        each parameter name to its value, min, max and unit. Additive models
        get an extra "norm" parameter, which nDspec applies itself.
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
                par_name, par_info = _parse_parameter(line)
            except (ValueError, IndexError) as exc:
                warnings.warn(f"{model_name}: {exc}", UserWarning)
                continue
            parameters[par_name] = par_info

        if model_type == "add":
            parameters["norm"] = {"value": 1.0, "min": 0.0, "max": 1e20, "unit": "n/a"}
        models_info[model_name] = {"func_call": func_call,
                                   "type": model_type,
                                   "parameters": parameters}
    return models_info


def _xspec_call(func_call, lib):
    """
    xspectrampoline callable ``f(ear, params, flux, err)`` and its array dtype
    for a model.dat function field, looked up in the ctypes library `lib`.
    The prefix gives the routine's language (Xspec manual, "Writing a new
    model function"); a missing routine raises AttributeError.
    """
    prefix, base = (func_call[:2], func_call[2:]) if func_call[:2] in ("C_", "c_", "F_") else ("", func_call)
    symbol, interface, dtype = {
        "C_": (func_call, "c", np.float64),            # C++, through its extern "C" wrapper
        "c_": (base, "c", np.float64),                 # C
        "F_": (base.lower() + "_", None, np.float64),  # Fortran, double precision
        "":   (base.lower() + "_", None, np.float32),  # Fortran, single precision
    }[prefix]
    return _XS.get_model(symbol, interface=interface, lib=lib), dtype


class ModelInterface():
    """
    Evaluate Xspec models, and models from Xspec-compatible local libraries
    (e.g. Relxill), as Python methods.

    The Xspec models are always available. Local model libraries can be added
    at any time with add_library. Models are then made available by name with
    add_model, and evaluated as ``lib.<name>(ear, params)`` (``lib.<name>(ear,
    params, spectrum)`` for convolution models), where ``ear`` holds the
    energy bin edges in keV. Additive models return photons/cm^2/s/keV,
    including the normalisation (the last parameter); multiplicative and
    convolution models return what the Xspec model computes.

    Attributes:
    -----------
    libraries: dict
        {library name: {"path", "pars_path", "models"}} for the Xspec library
        (name "xspec") and every library added with add_library; "models" is
        the parsed model file.

    models_info: dict
        Information on the models added with add_model: type, library, and
        the value, minimum, maximum and unit of each parameter, e.g.
        models_info['nthcomp']['parameters']['kT_e']['unit'] == "keV".
    """
    def __init__(self):
        self.libraries = {}
        self._libs = {}          # library name -> ctypes.CDLL
        self.models_info = {}
        self._register("xspec", _XS.lib_xs_functions, _find_model_dat())

    def _register(self, name, cdll, pars_path):
        self._libs[name] = cdll
        self.libraries[name] = {"path": cdll._name, "pars_path": str(pars_path),
                                "models": parse_model_file(pars_path)}

    def add_library(self, lib_path, pars_path, name=None):
        """
        Add an Xspec-compatible local model library and its lmodel.dat. Models 
        whose names also exist in another library are not added automatically: 
        add_model asks for the library explicitly.

        Parameters:
        -----------
        lib_path: str
            The compiled model library (.so on Linux, .dylib on MacOS).

        pars_path: str
            Its model description file (lmodel.dat).

        name: str, optional
            Name to refer to the library by, e.g. in add_model(...,
            library=name). Defaults to the file name without "lib" and the
            extension, e.g. "relxill" for librelxill.so.
        """
        if name is None:
            name = os.path.splitext(os.path.basename(str(lib_path)))[0]
            if name.startswith("lib") and len(name) > 3:
                name = name[3:]
        if name in self.libraries:
            raise ValueError(f"A library named '{name}' is already loaded; pass name=...")
        # Loaded after xspectrampoline, so the Xspec utility functions a local
        # library calls resolve to the ones xspectrampoline loaded.
        self._register(name, ct.CDLL(str(lib_path), mode=ct.RTLD_GLOBAL), pars_path)
        return name

    def available_models(self, library=None):
        """
        {model name: [libraries defining it]} for every model that can be
        added, optionally restricted to one library.
        """
        out = {}
        for libname, entry in self.libraries.items():
            if library is None or libname == library:
                for model in entry["models"]:
                    out.setdefault(model, []).append(libname)
        return out

    def _find(self, model_name, library):
        """
        Searches for the library defining a given model.
        """
        if library is not None:
            if library not in self.libraries:
                raise KeyError(f"No library named '{library}'; loaded: {list(self.libraries)}")
            if model_name not in self.libraries[library]["models"]:
                raise KeyError(f"Model '{model_name}' is not defined in "
                               f"{self.libraries[library]['pars_path']}")
            return library
        found = self.available_models().get(model_name, [])
        if not found:
            raise KeyError(f"Model '{model_name}' is not defined in any loaded library "
                           f"({', '.join(self.libraries)})")
        if len(found) > 1:
            raise ValueError(f"Model '{model_name}' is defined in several libraries "
                             f"({', '.join(found)}); choose one with library=...")
        return found[0]

    def _add_model(self, model_name, library=None):
        """
        Make a model available as a method of this object:

            lib._add_model("powerlaw")
            flux = lib.powerlaw(ear, [gamma, norm])

        Parameters:
        -----------
        model_name: str
            The model name, as in model.dat or lmodel.dat (case-insensitive).

        library: str, optional
            The library to take the model from. Only needed when more than one
            loaded library defines a model with this name.
        """
        model_name = model_name.lower()
        library = self._find(model_name, library)
        info = dict(self.libraries[library]["models"][model_name])
        info["library"] = library
        model_type = info["type"]
        if model_type not in _SUPPORTED_TYPES:
            raise NotImplementedError(
                f"Model type '{model_type}' ({model_name}) is not supported.")

        lib_func, dtype = _xspec_call(info["func_call"], self._libs[library])
        self.models_info[model_name] = info

        def prepare(ear, params):
            # check bounds before any cast to float32, so that values at a
            # limit are not pushed across it by rounding
            in_bounds = self.check_param_values(model_name, np.asarray(params, dtype=np.float64))
            ear = np.ascontiguousarray(ear, dtype=dtype)
            params = np.ascontiguousarray(params, dtype=dtype)
            if ear.ndim != 1 or ear.size < 2:
                raise ValueError("ear must be a 1D array of at least 2 bin edges")
            return in_bounds, ear, params

        if model_type == "add":
            def wrapper(ear, params):
                in_bounds, ear, params = prepare(ear, params)
                if not in_bounds:
                    return np.full(len(ear) - 1, np.nan)
                # the normalisation is applied here, not passed to the model
                flux = np.zeros(len(ear) - 1, dtype=dtype)
                lib_func(ear, np.ascontiguousarray(params[:-1]), flux, np.zeros_like(flux))
                return (flux / np.diff(ear)).astype(np.float64) * float(params[-1])
        elif model_type == "mul":
            def wrapper(ear, params):
                in_bounds, ear, params = prepare(ear, params)
                if not in_bounds:
                    return np.full(len(ear) - 1, np.nan)
                flux = np.zeros(len(ear) - 1, dtype=dtype)
                lib_func(ear, params, flux, np.zeros_like(flux))
                return flux.astype(np.float64)
        else:   # con: the input spectrum is modified in place
            def wrapper(ear, params, seed):
                in_bounds, ear, params = prepare(ear, params)
                if not in_bounds:
                    return np.full(len(ear) - 1, np.nan)
                flux = np.array(seed, dtype=dtype, copy=True)
                if flux.shape != (len(ear) - 1,):
                    raise ValueError("seed must have len(ear)-1 elements")
                lib_func(ear, params, flux, np.zeros_like(flux))
                return flux.astype(np.float64)

        wrapper.__name__ = model_name
        wrapper.__doc__ = (f"Xspec {model_type} model '{model_name}' ({library}); "
                           f"parameters: {', '.join(info['parameters'])}.")
        setattr(self, model_name, wrapper)
        return wrapper

    def add_models(self, *models, library=None):
        """Add several models at once: add_models("tbabs", "diskbb")."""
        for model in models:
            self._add_model(model, library=library)

    def check_models(self):
        """
        {library: [models]} for models listed in a model file whose routine
        cannot be found in the library (without calling anything).
        """
        missing = {}
        for libname, entry in self.libraries.items():
            for model, info in entry["models"].items():
                if info["type"] in _SUPPORTED_TYPES:
                    try:
                        _xspec_call(info["func_call"], self._libs[libname])
                    except AttributeError:
                        missing.setdefault(libname, []).append(model)
        return missing

    def print_model_info(self):
        """
        Print the models added with add_model, with their type, library and
        parameter names, default/min/max values and units.
        """
        print()
        print("Initialized Xspec models:")
        for component, details in self.models_info.items():
            print(f"{component}:")
            print(f"  type: {details['type']}")
            print(f"  library: {details['library']}")
            print("  parameters:")
            for param, values in details['parameters'].items():
                value_str = ', '.join(f"{key}: {value}" for key, value in values.items())
                print(f"    {param}: {value_str}")
            print()

    def check_param_values(self, model_name, params):
        """
        Check that the number of parameters is right and each value is within
        its hard limits; returns False (with a warning) otherwise, in which
        case the model evaluation returns NaN.
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


class _LegacyInterface(ModelInterface):
    _default_library = None

    def add_model(self, model, library=None, **ignored):
        # the old API took a dummy function named after the model, and only
        # looked in the library the object was created with
        name = model if isinstance(model, str) else model.__name__.rstrip("_")
        return super().add_model(name, library=library or self._default_library)

    def load_models(self, models, library=None):
        for model in (models.values() if isinstance(models, dict) else models):
            self.add_model(model, library=library)


class FortranInterface(_LegacyInterface):
    """Deprecated: use ModelInterface()."""
    def __init__(self, lib_path=None, pars_path=None, **ignored):
        warnings.warn("FortranInterface is deprecated; use ModelInterface() "
                      "(and add_library for local models).", DeprecationWarning, stacklevel=2)
        super().__init__()
        if lib_path is not None:
            self._default_library = self.add_library(lib_path, pars_path)


class CInterface(_LegacyInterface):
    """Deprecated: use ModelInterface().add_library(lib_path, pars_path)."""
    def __init__(self, lib_path, pars_path, **ignored):
        warnings.warn("CInterface is deprecated; use ModelInterface() and "
                      "add_library(lib_path, pars_path).", DeprecationWarning, stacklevel=2)
        super().__init__()
        self._default_library = self.add_library(lib_path, pars_path)
