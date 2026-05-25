from types import SimpleNamespace

import numpy as np

import go2_simulation.simulate_go2 as sim


def test_foot_contact_with_world_geom_counts_as_support():
    sim.np = np
    model = SimpleNamespace(geom_bodyid=np.asarray([0, 2, 0], dtype=np.int32))
    data = SimpleNamespace(
        ncon=1,
        contact=[
            SimpleNamespace(
                dist=0.0,
                geom1=1,
                geom2=2,
            )
        ],
    )
    foot_geom_ids = {"FL": 1, "FR": -1, "RL": -1, "RR": -1}
    force_sensor_ids = {leg: -1 for leg in sim.LEGS}

    contacts, all_feet_contact = sim._read_foot_contacts(
        model, data, floor_geom_id=0, foot_geom_ids=foot_geom_ids, force_sensor_ids=force_sensor_ids
    )

    assert contacts["FL"] is True
    assert all_feet_contact is False


def test_foot_contact_with_robot_body_does_not_count_as_support():
    sim.np = np
    model = SimpleNamespace(geom_bodyid=np.asarray([0, 2, 5], dtype=np.int32))
    data = SimpleNamespace(
        ncon=1,
        contact=[
            SimpleNamespace(
                dist=0.0,
                geom1=1,
                geom2=2,
            )
        ],
    )
    foot_geom_ids = {"FL": 1, "FR": -1, "RL": -1, "RR": -1}
    force_sensor_ids = {leg: -1 for leg in sim.LEGS}

    contacts, _ = sim._read_foot_contacts(
        model, data, floor_geom_id=0, foot_geom_ids=foot_geom_ids, force_sensor_ids=force_sensor_ids
    )

    assert contacts["FL"] is False
