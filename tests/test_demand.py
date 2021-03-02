import pytest
import numpy as np
from plan_opt.demand import Demand

from assets.demand.sample_demand import sample_demand
from plan_opt.demand_small_samples import four_weeks_uprising as sample_demand


def create_demand():
    demand = Demand(period=len(sample_demand), data=sample_demand)
    return demand


def test_demand_instantiation():
    d = create_demand()
    assert isinstance(d, Demand)
    np.testing.assert_array_equal(d.data, sample_demand)
