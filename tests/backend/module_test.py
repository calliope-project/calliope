"""Test backend module functionality (`__init__.py`)."""

import pytest

from calliope import AttrDict, backend
from calliope.backend.backend_model import BackendModel
from calliope.exceptions import BackendError


@pytest.mark.parametrize("valid_backend", ["pyomo", "gurobi"])
def test_valid_model_backend(simple_supply, valid_backend):
    """Requesting a valid model backend must result in a backend instance."""
    build_config = simple_supply.config.build.update({"backend": valid_backend})
    backend_obj = backend.get_model_backend(
        build_config, simple_supply.inputs, simple_supply.math.build
    )
    assert isinstance(backend_obj, BackendModel)


@pytest.mark.parametrize("spam", ["not_real", None, True, 1])
def test_invalid_model_backend(spam, simple_supply):
    """Backend requests should catch invalid setups."""
    invalid_config = AttrDict({"backend": spam})
    with pytest.raises(BackendError):
        backend.get_model_backend(
            invalid_config, simple_supply.inputs, simple_supply.math.build
        )
