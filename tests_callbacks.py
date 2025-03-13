from app import create_params_from_selectors, check_inputs_validity, \
    update_graph, update_county_selector, update_school_selector, get_school_vax_rate, \
    SELECTOR_DEFAULTS
import pytest

from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict

all_zero_params = {
    "I0": [0],
    "vax_prop": [0.0],
    "population": [0],
}

zero_I0_all_vax = {
    "I0": [0],
    "vax_prop": [1],
    "population": [int(1e6)],
}

just_enough_unvaccinated = {
    "I0": [5],
    "vax_prop": [0.95],
    "population": [100],
}

too_many_I0 = {
    "I0": [80],
    "vax_prop": [0.5],
    "population": [100],
}

warning_str = "Invalid inputs: The number of initially infected students " \
              "cannot exceed the number of unvaccinated students. Please adjust."

INPUTS_FAIL = (False, warning_str)
INPUTS_PASS = (True, "")


@pytest.mark.parametrize("test_input, expected_output",
                         [(all_zero_params, INPUTS_PASS),
                          (zero_I0_all_vax, INPUTS_PASS),
                          (just_enough_unvaccinated, INPUTS_PASS),
                          (too_many_I0, INPUTS_FAIL)])
def test_check_inputs_validity(test_input, expected_output):
    assert check_inputs_validity(test_input) == expected_output


input_1 = {
    "params_dict": {},
    "school_size": 100,
    "vax_rate_percent": 95,
    "I0": 10,
    "R0": 3,
    "latent_period": 10,
    "infectious_period": 5
}


@pytest.mark.parametrize("test_input", [input_1])
def test_create_params_from_selectors(test_input):

    params = create_params_from_selectors(**test_input)[0]

    assert len(params["population"]) == 1
    assert len(params["vax_prop"]) == 1
    assert len(params["I0"]) == 1

    assert 1 >= params["vax_prop"][0] > 1e-6


# Reproducibility of graph
# Maybe check formatting