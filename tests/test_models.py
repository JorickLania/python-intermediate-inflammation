"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

def test_daily_min_string():
    """Test for a TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'],['General','Kenobi']])


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)

def test_daily_mean_floats():
    """Test that mean function works for an array of positive floats."""
    from inflammation.models import daily_mean

    test_input = np.array([[1.5, 2.5],
                           [3.5, 4.5],
                           [5.5, 6.5]])
    test_result = np.array([3.5, 4.5])

    npt.assert_array_equal(daily_mean(test_input), test_result)

def test_daily_max_integers():
    """Test that daily max function works for array of positive integers"""
    from inflammation.models import daily_max

    test_input = np.array([[4, 2, 5],
                           [10, 5, 4],
                           [3, 6, 7]])
    test_result = np.array([10, 6, 7])

    npt.assert_array_equal(daily_max(test_input),test_result)

def test_daily_min_negative_integers():
    """Test that daily min function works for array of positive and negative
       integers."""
    from inflammation.models import daily_min

    test_input = np.array([[4, -2, 5],
                           [-1, 3, -9],
                           [7, 8, -2]])
    test_result = np.array([-1, -2, -9])

    npt.assert_array_equal(daily_min(test_input), test_result)


