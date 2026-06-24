import numpy as np

from gardener_bdx.common import math_utils as m


def test_quat_identity_rotates_nothing():
    q = np.array([1.0, 0, 0, 0])
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(m.quat_rotate(q, v), v)


def test_euler_quat_roundtrip():
    for rpy in [(0.1, -0.2, 0.3), (0.0, 0.0, 1.5), (-0.4, 0.2, -2.0)]:
        q = m.quat_from_euler(*rpy)
        assert np.allclose(m.euler_from_quat(q), rpy, atol=1e-6)


def test_projected_gravity_upright():
    q = np.array([1.0, 0, 0, 0])
    assert np.allclose(m.projected_gravity(q), [0, 0, -1], atol=1e-6)


def test_upright_cosine_tipping():
    assert m.upright_cosine(m.quat_from_euler(0, 0, 0)) > 0.99
    # Rolled 90 degrees -> on its side -> ~0.
    assert abs(m.upright_cosine(m.quat_from_euler(np.pi / 2, 0, 0))) < 1e-6


def test_inverse_rotation_consistency():
    q = m.quat_from_euler(0.2, 0.3, -0.4)
    v = np.array([0.5, -1.0, 2.0])
    assert np.allclose(m.quat_rotate_inverse(q, m.quat_rotate(q, v)), v, atol=1e-6)
