# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name, no-self-use, unused-argument, abstract-class-instantiated, invalid-name, pointless-statement, protected-access
import numpy as np
import pytest

from symaware.base.data.data_structures import AwarenessVector, TimeSeries


class TestAwarenessVector:
    def test_awareness_vector_init(self):
        ID = 1
        current_state = np.zeros(10)
        intent = TimeSeries({0: np.ones(10)})
        risk = TimeSeries({0: np.zeros(10)})
        uncertainty = TimeSeries({0: np.zeros(10)})

        awareness_vector = AwarenessVector(ID, current_state, intent, risk, uncertainty)

        assert awareness_vector.ID == ID
        assert np.array_equal(awareness_vector.state, current_state)
        assert awareness_vector.intent == intent
        assert awareness_vector.risk == risk
        assert awareness_vector.uncertainty == uncertainty

    def test_awareness_vector_state_setter(self):
        awareness_vector = AwarenessVector(1, np.zeros(10))
        new_state = np.ones(10)

        awareness_vector.state = new_state

        assert np.array_equal(awareness_vector.state, new_state)

    def test_awareness_vector_intent_setter(self):
        awareness_vector = AwarenessVector(1, np.zeros(10))
        new_intent = TimeSeries({0: np.ones(10)})

        awareness_vector.intent = new_intent

        assert awareness_vector.intent == new_intent

    def test_awareness_vector_risk_setter(self):
        awareness_vector = AwarenessVector(1, np.zeros(10))
        new_risk = TimeSeries({0: np.zeros(10)})

        awareness_vector.risk = new_risk

        assert awareness_vector.risk == new_risk

    def test_awareness_vector_uncertainty_setter(self):
        awareness_vector = AwarenessVector(1, np.zeros(10))
        new_uncertainty = TimeSeries({0: np.zeros(10)})

        awareness_vector.uncertainty = new_uncertainty

        assert awareness_vector.uncertainty == new_uncertainty

    def test_awareness_vector_copy(self):
        awareness_vector = AwarenessVector(1, np.zeros(10))
        awareness_vector.intent = TimeSeries({0: np.ones(10)})
        awareness_vector.risk = TimeSeries({0: np.zeros(10)})
        awareness_vector.uncertainty = TimeSeries({0: np.zeros(10)})

        copied_vector = awareness_vector.copy()

        assert copied_vector == awareness_vector
        assert copied_vector is not awareness_vector

    def test_awareness_vector_or_operator(self):
        awareness_vector1 = AwarenessVector(1, np.zeros(10))
        awareness_vector1.intent = TimeSeries({0: np.ones(10)})
        awareness_vector1.risk = TimeSeries({0: np.zeros(10)})
        awareness_vector1.uncertainty = TimeSeries({0: np.zeros(10)})

        awareness_vector2 = AwarenessVector(1, np.ones(10))
        awareness_vector2.intent = TimeSeries({0: np.zeros(10)})
        awareness_vector2.risk = TimeSeries({1: np.ones(10)})
        awareness_vector2.uncertainty = TimeSeries({0: np.zeros(15)})

        result = awareness_vector1 | awareness_vector2

        assert result.ID == awareness_vector1.ID
        assert np.array_equal(result.state, awareness_vector2.state)
        assert result.intent == (awareness_vector1.intent | awareness_vector2.intent)
        assert result.risk == (awareness_vector1.risk | awareness_vector2.risk)
        assert result.uncertainty == (awareness_vector1.uncertainty | awareness_vector2.uncertainty)

    def test_awareness_vector_equal_operator(self):
        awareness_vector1 = AwarenessVector(1, np.zeros(10))
        awareness_vector1.intent = TimeSeries({0: np.ones(10)})
        awareness_vector1.risk = TimeSeries({0: np.zeros(10)})
        awareness_vector1.uncertainty = TimeSeries({0: np.zeros(10)})

        awareness_vector2 = AwarenessVector(1, np.zeros(10))
        awareness_vector2.intent = TimeSeries({0: np.ones(10)})
        awareness_vector2.risk = TimeSeries({0: np.zeros(10)})
        awareness_vector2.uncertainty = TimeSeries({0: np.zeros(10)})

        assert awareness_vector1 == awareness_vector2

    def test_awareness_vector_not_equal_operator(self):
        awareness_vector1 = AwarenessVector(1, np.zeros(10))
        awareness_vector1.intent = TimeSeries({0: np.ones(10)})
        awareness_vector1.risk = TimeSeries({0: np.zeros(10)})
        awareness_vector1.uncertainty = TimeSeries({0: np.zeros(10)})

        awareness_vector2 = AwarenessVector(2, np.zeros(10))
        awareness_vector2.intent = TimeSeries({0: np.ones(10)})
        awareness_vector2.risk = TimeSeries({0: np.zeros(10)})
        awareness_vector2.uncertainty = TimeSeries({0: np.zeros(10)})

        assert awareness_vector1 != awareness_vector2

    def test_awareness_vector_str(self):
        awareness_vector = AwarenessVector(1, np.zeros(10))
        awareness_vector.intent = TimeSeries({0: np.ones(10)})
        awareness_vector.risk = TimeSeries({0: np.zeros(10)})
        awareness_vector.uncertainty = TimeSeries({0: np.zeros(10)})

        expected_output = (
            "Agent ID: 1\n"
            "Current State: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n"
            "Intent: {0: array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])}\n"
            "Risk: {0: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])}\n"
            "Uncertainty: {0: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])}\n"
        )

        assert str(awareness_vector) == expected_output


class TestTimeSeries:
    def test_timeseries_init_empty(self):
        ts = TimeSeries()
        assert len(ts) == 0

    def test_timeseries_init_dict(self):
        data = {0: np.zeros(10), 1: np.ones(10)}
        ts = TimeSeries(data)
        assert len(ts) == 2
        assert np.array_equal(ts[0], np.zeros(10))
        assert np.array_equal(ts[1], np.ones(10))

    def test_timeseries_init_invalid(self):
        invalid_data = {0: "invalid", 1: np.ones(10)}
        with pytest.raises(TypeError):
            TimeSeries(invalid_data)

    def test_timeseries_setitem(self):
        ts = TimeSeries()
        data = np.ones(10)
        ts[0] = data
        assert np.array_equal(ts[0], data)

    def test_timeseries_setitem_invalid(self):
        # Test setting item with invalid data types
        ts = TimeSeries()
        with pytest.raises(TypeError):
            ts["invalid"] = "invalid data"

    def test_timeseries_getitem(self):
        ts = TimeSeries({0: np.zeros(10), 1: np.ones(10)})
        assert np.array_equal(ts[0], np.zeros(10))
        assert np.array_equal(ts[1], np.ones(10))

    def test_timeseries_getitem_invalid(self):
        ts = TimeSeries()
        with pytest.raises(KeyError):
            ts[2]

    def test_timeseries_eq_operator(self):
        ts1 = TimeSeries({0: np.zeros(10), 1: np.ones(10)})
        ts2 = TimeSeries({0: np.zeros(10), 1: np.ones(10)})
        ts3 = TimeSeries({0: np.zeros(10), 1: np.ones(5)})

        assert ts1 == ts2
        assert ts1 != ts3

    def test_timeseries_or_operator(self):
        ts1 = TimeSeries({0: np.zeros(10), 1: np.ones(10)})
        ts2 = TimeSeries({1: np.ones(5), 2: np.full(10, 2)})

        result = ts1 | ts2
        assert len(result) == 3
        assert np.array_equal(result[0], np.zeros(10))
        assert np.array_equal(result[1], np.ones(5))
        assert np.array_equal(result[2], np.full(10, 2))

    def test_timeseries_or_operator_invalid(self):
        ts = TimeSeries()
        with pytest.raises(TypeError):
            ts | "invalid"
