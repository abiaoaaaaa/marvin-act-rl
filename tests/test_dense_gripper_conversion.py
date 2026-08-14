import numpy as np

from convert_gripper_to_dense_target import densify_episode_arrays


def test_densify_gripper_commands_and_last_action_state():
    actions = np.zeros((7, 5), dtype=np.float32)
    actions[:, 4] = [0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    states = np.zeros((7, 24), dtype=np.float32)
    states[0, 0] = 1.0  # physically open at episode start

    dense_actions, dense_states, details = densify_episode_arrays(actions, states)

    np.testing.assert_array_equal(dense_actions[:, 4], [1, -1, -1, -1, 1, 1, 1])
    np.testing.assert_array_equal(dense_states[:, 5], [1, 1, -1, -1, -1, 1, 1])
    np.testing.assert_array_equal(dense_actions[:, :4], actions[:, :4])
    assert details["initial_target"] == 1
    assert details["target_transitions"] == 2


def test_densify_starts_closed_and_uses_thresholds():
    actions = np.zeros((4, 5), dtype=np.float32)
    actions[:, 4] = [0.49, 0.5, -0.49, -0.5]
    states = np.zeros((4, 24), dtype=np.float32)
    states[0, 0] = 0.2

    dense_actions, dense_states, details = densify_episode_arrays(actions, states)

    np.testing.assert_array_equal(dense_actions[:, 4], [-1, 1, 1, -1])
    np.testing.assert_array_equal(dense_states[:, 5], [-1, -1, 1, 1])
    assert details["initial_target"] == -1
    assert details["target_transitions"] == 2
