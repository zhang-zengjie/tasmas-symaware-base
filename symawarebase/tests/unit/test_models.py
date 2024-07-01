# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name, no-self-use, unused-argument, abstract-class-instantiated, invalid-name
import numpy as np

from symaware.base.models import DynamicalModel


class TestDynamicalModel:
    def test_dynamical_model_init(self, PatchedDynamicalModel: type[DynamicalModel]):
        ID = 1
        control_input = np.zeros(2)
        model = PatchedDynamicalModel(ID, control_input)
        assert model is not None
        assert model.id == ID
        assert model.control_input is not None
        assert model.control_input_shape == control_input.shape
