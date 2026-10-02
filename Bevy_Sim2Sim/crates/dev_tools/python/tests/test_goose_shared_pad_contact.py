"""Shared compression, original stroke and one native integration are physical invariants."""
import mujoco
import numpy as np
import pytest

from bevy_microduck_tools.goose.shared_pad_contact import SharedPad, shared_step, solve_shared_pads


@pytest.fixture(autouse=True)
def outside_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_two_points_share_one_slide_and_unloaded_corner_cannot_pull():
    pad = SharedPad(1, 'patch', 1000., 0., .01)
    result = solve_shared_pads([[2.]], [-.2], [[1.], [1.]], [-.001, -.0005],
                               [0, 0], [1., 1.], [pad], [0.])
    expected_impulse = .005/(.02/2.+1./(.02*1000.))
    assert result['normal_impulses'][0] == pytest.approx(expected_impulse, rel=2e-7)
    assert result['normal_impulses'][1] == 0.
    assert result['compression'][0] == pytest.approx(expected_impulse/(.02*1000.), rel=2e-7)
    assert result['velocity'][0] == pytest.approx(-.2+expected_impulse/2., rel=2e-7)


def test_release_retains_original_damping_and_shared_compression_history():
    pad = SharedPad(1, 'patch', 1000., 2., .01)
    prior = np.array([.001])
    result = solve_shared_pads([[1.]], [0.], np.empty((0,1)), [], [], [], [pad], prior)
    assert result['compression'][0] == pytest.approx(2.*.001/(2.+.02*1000.))
    assert result['velocity'][0] == 0.
    assert result['normal_impulses'].size == 0
    assert .5*1000.*result['compression'][0]**2 < .5*1000.*prior[0]**2
    assert prior[0] == .001


def test_upper_stroke_reaction_participates_in_the_same_solve():
    pad = SharedPad(1, 'patch', 23357.0304, 2., .0015)
    result = solve_shared_pads([[.293]], [-10.], [[1.]], [0.], [0], [1.], [pad], [0.])
    assert result['upper_stop_impulses'][0] > 0.
    assert result['compression'][0] == pytest.approx(.0015, abs=2e-8)
    assert result['max_travel_violation_m'] < 2e-8
    assert result['velocity'][0] > -10.


def test_coupled_physical_coordinates_do_not_duplicate_contact_mass():
    pad = SharedPad(1, 'patch', 1000., 0., .01)
    mass = np.array([[2., .2], [.2, .4]])
    row = np.array([[1., .3]])
    result = solve_shared_pads(mass, [-.2, .1], row, [-.001], [0], [1.], [pad], [0.])
    expected = np.array([-.2, .1])+np.linalg.solve(mass,row[0])*result['normal_impulses'][0]
    np.testing.assert_allclose(result['velocity'], expected, atol=1e-14)


def fixture(frictionless=True):
    patches = [(x,y) for x in [-.08, 0., .08] for y in [-.025, .025]]
    geometry = ''.join(f'<geom name="p{i}" type="box" pos="{x} {y} 0" size=".002 .01 .001"/>'
                       for i,(x,y) in enumerate(patches))
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".02" integrator="Euler" gravity="0 0 0">
        <flag autoreset="disable" eulerdamp="disable"/>
      </option><default><geom condim="{1 if frictionless else 3}" friction="0 0 0" margin=".1"/></default>
      <worldbody><geom type="plane" size="1 1 .1"/>
        <body pos="0 0 .0015"><inertial mass=".293" pos=".01 0 -.005" diaginertia=".001 .002 .003"/>
          <freejoint/>{geometry}</body></worldbody></mujoco>''')
    pads = tuple(SharedPad(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,f'p{i}'),
                          f'p{i}',23357.0304,2.,.0015) for i in range(6))
    return model, mujoco.MjData(model), pads


def test_shared_fixture_uses_exactly_one_native_euler_and_no_step2(monkeypatch):
    model, data, pads = fixture()
    before = data.qpos.copy()
    data.xfrc_applied[1,:3] = [0.,0.,-30.]
    counters = {'mj_step1':0, 'mj_step2':0, 'mj_Euler':0}
    for name in counters:
        native = getattr(mujoco,name)
        def count(m,d,native=native,name=name):
            counters[name] += 1
            native(m,d)
        monkeypatch.setattr(mujoco,name,count)
    result, rows = shared_step(model,data,pads,np.zeros(6))
    assert counters == {'mj_step1':1, 'mj_step2':0, 'mj_Euler':1}
    assert data.time == pytest.approx(.02)
    np.testing.assert_allclose(data.qvel,result['velocity'],atol=1e-14)
    np.testing.assert_allclose(data.qpos[:3],before[:3]+.02*data.qvel[:3],atol=1e-14)
    actual_force = sum(r['normal_force_n'] for r in rows)
    mass = np.zeros((6,6))
    mujoco.mj_fullM(model,data,mass)
    # The free-joint origin differs from COM; use the complete physical M row.
    assert mass[2]@data.qacc == pytest.approx(actual_force-30.,abs=1e-10)
    assert not any(w.number for w in data.warning)


def test_friction_rejected_before_native_integration():
    model,data,pads = fixture(False)
    with pytest.raises(ValueError, match='frictionless'):
        shared_step(model,data,pads,np.zeros(6))
    assert data.time == 0.


def test_articulated_runtime_rejected_even_though_algebra_supports_generalized_coordinates():
    model,data,pads = fixture()
    model.jnt_type[0] = int(mujoco.mjtJoint.mjJNT_SLIDE)
    with pytest.raises(ValueError,match='isolated six-DoF'):
        shared_step(model,data,pads,np.zeros(6))
    assert data.time == 0.
