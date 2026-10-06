import copy

import pytest

from bevy_microduck_tools.goose.artifacts import JOINT_ORDER
from bevy_microduck_tools.goose.mature_training import native_physical_action_std


def si_contract():
    return {"joint_order": list(JOINT_ORDER), "joints": [
        {"name": name, "torque_peak_limit_nm": 12., "kp_nm_rad": 100.,
         "action_scale_rad": 2.} for name in JOINT_ORDER]}


def test_same_physical_exploration_in_different_public_action_units():
    contract = si_contract()
    contract["joints"][1]["action_scale_rad"] = .5
    before = copy.deepcopy(contract)
    std = native_physical_action_std(contract)
    # Upstream requests .03rad / 3Nm for this 12Nm,100Nm/rad actuator.
    assert std[0] == pytest.approx(.015)
    assert std[1] == pytest.approx(.06)
    assert std[0]*2 == pytest.approx(.03)
    assert std[1]*.5 == pytest.approx(.03)
    assert len(std) == 18 and contract == before


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_invalid_stiffness_cannot_silently_initialize_policy(value):
    contract = si_contract()
    contract["joints"][8]["kp_nm_rad"] = value
    with pytest.raises(ValueError, match="positive finite SI"):
        native_physical_action_std(contract)


def test_reordered_axes_are_rejected():
    contract = si_contract()
    contract["joints"][6], contract["joints"][7] = (
        contract["joints"][7], contract["joints"][6])
    with pytest.raises(ValueError, match="original18-axis order"):
        native_physical_action_std(contract)


def test_overflow_cannot_produce_zero_gaussian_std():
    contract = si_contract()
    contract["joints"][8]["kp_nm_rad"] = 1e308
    contract["joints"][8]["action_scale_rad"] = 1e308
    with pytest.raises(ValueError, match="finite positive std"):
        native_physical_action_std(contract)
