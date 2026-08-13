# Code formatting

1. Code lines should not be longer than 120 columns, and strive to not exceed 80 columns.
2. Indents are always 4 spaces.
3. Variables should explain what they are. You should be able to understand what the variable is for at a glance.
4. Functions, classes and methods should contain a docstring describing what the code does, as well as the arguments in input and output. Similarly, classes should describe every attribute stored or inherited.
5. Independent variable arrays should use the same consistent format. They should be plural, and each independent variable should be named consistently throughout the library: 
    * Arrays of time should be named times
    * Arrays of Fourier frequency should be named "freqs"
    * Arrays of physical energy bin centers should be named "energs"
    * Arrays of energy bin edges should be named "ear", identically to the Xspec convention
    * Arrays of energy channel edges should be named "emin" or "emax", identically to the OGIP format 
    * Arrays of modulation angles should be named "mod_angles"
6. Class names use Pascal case to aggregate words - e.g. SpectralTimingClass, not spectral_timing_class
7. Method names use snake case to aggregate words, and strive to use complete words where possible - e.g. set_polarimetry_value, not setPolarimetryValue 
8. Methods meant for internal book-keeping and computations, rather than for user-facing tasks, should star with an underscore - e.g. self._internal_calculation, not self.internal_calculation 
9. All functions and methods should have a return line at their end; if there is no object to return, then
    ``` python
        blablabla code  
        return
    ```
   suffices. In class initializers, ```pass``` can replace ```return```.
 
# Doc strings
Doc strings are provided within each class, as well as each method or function. Doc strings 
should not include change logs. They should first describe what the method/function/class does,
and then list input and output as appropriate. 

```python
def blah(arguments)
    """
    This function does things 
    
    Parameters:
    -----------
    arguments: type of the argument 
        This argument is something
        
    Output:
    -------
    some_return: type of the return 
        This is what the function returns 
    """
    code
    
    return some_return
```

# Structure formatting

1. Classes should be divided between ```Operators```, which execute any necessary model computation, and ```FitterObjects```, which (also) utilize ```Operators``` to fit and visualize data. 
2. ```Operator``` classes are divided in files that each correspond to a concept - e.g. Timing.py, Polarimetry.py, ResponseMatrix.py, etc
3. ```FitterObjects``` should each be set in their separate file. They are all inherited from the ```SimpleFit``` class.
4. Functionality that does not belong to the above groups can go in dedicated files - e.g. XspecInterface.py for calling Xspec models, or SamplingUtils.py for interfacing with sampling algorithms.
5. The file naming convention is Pascal case, identically to classes - e.g. FitCrossSpectrum.py, not fit_cross_spectrum.py.

# Repository rules

1. Direct commits to main are not allowed. Changes to the main branch should only ever come from pull requests from a stand-alone developement branch.
2. Pull requests to main that include entirely new features should only be accepted if the new feature is included in a stand-alone tutorial notebook in the documentation, and if the documentation (with the new tutorial) builds correctly.
3. Pull requests with new features should not be accepted if those new features are not covered by unit tests. 
4. New releases (both on github and on PyPi) trigger automatically upon pushing a tag to the main branch. The correct format for each tag is ```v*```, where ```*``` is the version number (e.g. ```v0.5.3```). 
