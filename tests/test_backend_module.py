"""Test backend module functionality (`__init__.py`)."""

import pytest
from calliope import backend
from calliope.backend.backend_model import BackendModel, BackendSetup
from calliope.exceptions import BackendError


@pytest.mark.parametrize("valid_backend", backend.MODEL_BACKENDS)
def test_valid_model_backend(simple_supply, valid_backend):
    """Requesting a valid model backend must result in a backend instance."""
    setup = BackendSetup(inputs=simple_supply._model_data, math=simple_supply.math)
    backend_obj = backend.get_model_backend(valid_backend, setup)
    assert isinstance(backend_obj, BackendModel)


@pytest.mark.parametrize("spam", ["not_real", None, True, 1])
def test_invalid_model_backend(spam):
    """Backend requests should catch invalid backend names."""
    with pytest.raises(BackendError):
        backend.get_model_backend(spam, None)
