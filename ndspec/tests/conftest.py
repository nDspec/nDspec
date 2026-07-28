#backend to run plots in the CI unit tests on github
import matplotlib
matplotlib.use("Agg")

#skip xspec tests if heasoft is not installed - this is because 
#the tests require me to compare to the xspec output itself, meaning we 
#can't get around them with just xspectrampoline since we need the 
#software itself, which is awful. 
import pytest
def pytest_collection_modifyitems(config, items):
    try:
        import xspec 
        return
    except ImportError:
        skip_xspec = pytest.mark.skip(reason="PyXspec/HEASoft not installed")
        for item in items:
            if "xspec" in item.keywords:
                item.add_marker(skip_xspec)
