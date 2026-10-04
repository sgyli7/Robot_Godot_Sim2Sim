"""Actual frozen-source CPU kernel comparison; zero physics integration.

Set GOOSE_FROZEN_TASK_PROXY_PACKAGE to the independently verified 004 package.
The model stays in the project backup directory, not repository test assets.
"""
import os
from pathlib import Path

import pytest


def test_actual_frozen_source_contact_mapping(monkeypatch):
    package = os.environ.get('GOOSE_FROZEN_TASK_PROXY_PACKAGE')
    if not package:
        pytest.skip('Frozen external task-proxy package is required')
    np = pytest.importorskip('numpy')
    mujoco = pytest.importorskip('mujoco')
    mjw = pytest.importorskip('mujoco_warp')
    wp = pytest.importorskip('warp')
    package = Path(package).resolve(strict=True)
    monkeypatch.syspath_prepend(str(package / 'src'))
    from sai_agent.goose.task_proxy_runtime import TaskProxyRuntime
    from bevy_microduck_tools.goose.task_proxy_contact import FrozenContactAdapter

    runtime = TaskProxyRuntime(package / 'robots/Goose_V0.1/models/task_proxy_11_v1/robot.xml',
                               package / 'robots/Goose_V0.1/configs/task_proxy_11_v1_contract.json')
    m, d = runtime.model, runtime.data
    # The independently named compatibility candidate changes this option.
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    mujoco.mj_step1(m, d)
    with wp.ScopedDevice('cpu'):
        original_flags = int(m.opt.disableflags)
        try:
            m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
            wm = mjw.put_model(m)
        finally:
            m.opt.disableflags = original_flags
        wd = mjw.put_data(m, d, nconmax=64, njmax=128)
        adapter = FrozenContactAdapter(wm, wd, m, runtime.contract)
        adapter.apply()
        adapter.assert_valid()
        runtime._planar_sole_quadrature()
        J = wd.efc.J.numpy()[0]
        D = wd.efc.D.numpy()[0]
        scales = adapter.tangent_scale.numpy()
        actual = {n:getattr(wd.contact,n).numpy() for n in ('geom','pos','friction','efc_address')}
        expected = {(tuple(c.geom), tuple(np.round(c.pos,6))):c for c in d.contact
                    if c.efc_address >= 0 and runtime.ground in c.geom}
        assert len(expected) == int(wd.nacon.numpy()[0]) == 8
        assert not wm.is_sparse
        for i in range(8):
            key = (tuple(actual['geom'][i]), tuple(np.round(actual['pos'][i].astype(float),6)))
            c = expected[key]
            native_row = int(c.efc_address)
            rows = actual['efc_address'][i,:3]
            scale = scales[i]
            units = np.array([1.,scale,scale])
            native_J = np.asarray(d.efc_J).reshape(d.nefc,m.nv)[native_row:native_row+3]
            np.testing.assert_allclose(J[rows,:m.nv]*units[:,None],native_J,rtol=2e-5,atol=2e-6)
            np.testing.assert_allclose(actual['friction'][i,:2]/scale,c.friction[:2],rtol=2e-5)
            np.testing.assert_allclose(D[rows[1:]]/scale**2,d.efc_D[native_row+1:native_row+3],rtol=2e-5)
            expected_normal_D = .02 * (.02*140142.1824/4 + 12./4)
            np.testing.assert_allclose(D[rows[0]],expected_normal_D,rtol=2e-5)
        assert float(wd.time.numpy()[0]) == d.time == 0.
        assert runtime.physics_integrations == 0
