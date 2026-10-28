import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('__file__/ndspec/'))))

import numpy as np
import lmfit

import pytest

from ndspec import Utils

#model utilies
def _linear(x,slope=1.,intercept=0.):
    return slope*x + intercept
 
def _quadratic(x,curve=1.,offset=0.):
    return curve*x**2 + offset
 
def _make_model(prefix,slope=1.,intercept=0.):
    model = lmfit.Model(_linear,prefix=prefix)
    params = model.make_params(slope=slope,intercept=intercept)
    return model, params
 
def _merge_params(*param_sets):
    merged = param_sets[0].copy()
    for params in param_sets[1:]:
        merged.update(params)
    return merged

def test_contains_direct_leaf():
    a, pa = _make_model("a_")
    b, pb = _make_model("b_")
    composite = a+b
    assert Utils._contains(composite,a) is True
    assert Utils._contains(composite,b) is True
 
def test_contains_nested_leaf():
    a, pa = _make_model("a_")
    b, pb = _make_model("b_")
    c, pc = _make_model("c_")
    composite = a*(b+c)
    assert Utils._contains(composite,c) is True
 
def test_contains_missing_leaf():
    a, pa = _make_model("a_")
    b, pb = _make_model("b_")
    c, pc = _make_model("c_")
    composite = a+b
    assert Utils._contains(composite,c) is False
 
#_additive_leaves 
def test_ambiguous_product():
    a, pa = _make_model("a_")
    b, pb = _make_model("b_")
    c, pc = _make_model("c_")
    d, pd = _make_model("d_")
    with pytest.raises(ValueError):
        Utils._additive_leaves((a+b)*(c+d))
 
#_eval_term
def test_eval_term_isolates_additive_term():
    a, pa = _make_model("a_",slope=2.)
    b, pb = _make_model("b_",slope=3.)
    params = _merge_params(pa,pb)
    x = np.arange(1.,5.)
    values = Utils._eval_term(a+b,a,params,x=x)
    assert np.allclose(values,a.eval(params,x=x))
 
def test_eval_term_distributes_common_factor():
    a, pa = _make_model("a_",slope=2.)
    b, pb = _make_model("b_",slope=1.)
    c, pc = _make_model("c_",slope=1.)
    params = _merge_params(pa,pb,pc)
    x = np.arange(1.,5.)
    term_b = Utils._eval_term(a*(b+c),b,params,x=x)
    expected = a.eval(params,x=x)*b.eval(params,x=x)
    assert np.allclose(term_b,expected) 
 
#model_expand 
def test_single_component():
    model, params = _make_model("a_")
    x = np.arange(1.,5.)
    assert Utils.model_expand(model,params,x=x) == {}
 
def test_additive_components():
    a, pa = _make_model("a_",slope=2.)
    b, pb = _make_model("b_",slope=3.)
    params = _merge_params(pa,pb)
    x = np.arange(1.,5.)
    components = Utils.model_expand(a+b,params,x=x)
    assert set(components.keys()) == {"a","b"}
    assert np.allclose(components["a"],a.eval(params,x=x))
    assert np.allclose(components["b"],b.eval(params,x=x))
 
def test_no_distribute():
    a, pa = _make_model("a_",slope=2.)
    b, pb = _make_model("b_",slope=1.)
    c, pc = _make_model("c_",slope=1.)
    params = _merge_params(pa,pb,pc)
    x = np.arange(1.,5.)
    components = Utils.model_expand(a*(b+c),params,distribute=False,x=x)
    assert np.allclose(components["b"],b.eval(params,x=x))
    assert np.allclose(components["c"],c.eval(params,x=x))
 
def test_duplicate_labels():
    a = lmfit.Model(_linear,prefix="dup_")
    b = lmfit.Model(_quadratic,prefix="dup_")
    pa = a.make_params(slope=2.,intercept=0.)
    pb = b.make_params(curve=1.,offset=0.)
    params = _merge_params(pa,pb)
    x = np.arange(1.,5.)
    with pytest.raises(ValueError):
        Utils.model_expand(a+b,params,x=x)

#check_shape
def test_check_shape_mismatch_raises():
    with pytest.raises(ValueError):
        Utils.check_shape([1,2,3],4,"arr")

#check_matching_length
def test_check_matching_length_ignores_missing_arrays():
    Utils.check_matching_length("a",None,"b",np.arange(5))
    Utils.check_matching_length("a",np.arange(5),"b",None)
    Utils.check_matching_length("a",None,"b",None)

def test_check_matching_length_mismatch_raises():
    with pytest.raises(ValueError):
        Utils.check_matching_length("a",np.arange(5),"b",np.arange(4))

#check_two_arrays
def test_check_two_arrays_both_present_ok():
    Utils.check_two_arrays("a",np.arange(5),"b",np.arange(5))

def test_check_two_arrays_both_absent_ok():
    Utils.check_two_arrays("a",None,"b",None)

def test_check_two_arrays_only_first_raises():
    with pytest.raises(ValueError):
        Utils.check_two_arrays("a",np.arange(5),"b",None)

def test_check_two_arrays_only_second_raises():
    with pytest.raises(ValueError):
        Utils.check_two_arrays("a",None,"b",np.arange(5))
