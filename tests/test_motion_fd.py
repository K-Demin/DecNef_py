import numpy as np
import pytest

from motion_fd import fd_from_rtpspy_delta, fd_from_rtpspy_motion


def test_fd_uses_rotations_first_and_translations_last():
    delta = np.array([180.0, 0.0, 0.0, 1.0, 2.0, 3.0])

    fd = fd_from_rtpspy_delta(delta, radius_mm=10.0)

    assert fd == pytest.approx(10.0 * np.pi + 6.0)


def test_fd_time_series_starts_at_zero_and_differences_successive_trs():
    motion = np.array(
        [
            [10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
            [11.0, 18.0, 33.0, 5.0, 1.0, 9.0],
        ]
    )

    fd = fd_from_rtpspy_motion(motion, radius_mm=50.0)

    expected_second = 50.0 * np.deg2rad(1.0 + 2.0 + 3.0) + 4.0 + 1.0 + 6.0
    np.testing.assert_allclose(fd, [0.0, expected_second])


def test_fd_rejects_motion_with_wrong_number_of_columns():
    with pytest.raises(ValueError, match="T-by-6"):
        fd_from_rtpspy_motion(np.zeros((2, 5)))
