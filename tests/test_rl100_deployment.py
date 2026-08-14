import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import numpy as np

from lerobot_robot_marvin.backends import MockBackend
from lerobot_robot_marvin.errors import SafetyError
from lerobot_robot_marvin.rl100_policy import (
    MarvinObservationHistory,
    RL100_CAMERA_KEYS,
    sanitize_action_chunk,
)
from lerobot_robot_marvin.rl100_online import (
    MarvinBridgeOnlineEnv,
    MarvinOnlineReplayBuffer,
)
from lerobot_robot_marvin.rl100_rollout import (
    BridgeHeartbeat,
    RL100RolloutController,
    scale_motion_action,
    validate_bridge_preflight,
)
from lerobot_robot_marvin.rl100_replay import episode_bounds
from lerobot_robot_marvin.schema import validate_observation


class FakeClient:
    def __init__(self, success_at_step=3):
        self.backend = MockBackend(max_steps=20, success_at_step=success_at_step)
        self.backend.connect()
        self.actions = []

    def reset(self):
        observation, info = self.backend.reset()
        return validate_observation(observation), info

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        observation, reward, terminated, truncated, info = self.backend.step(action)
        return {
            "observation": validate_observation(observation),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "succeed": info["succeed"],
            "actual_action": np.asarray(action),
            "info": info,
        }


class FakePolicy:
    n_action_steps = 16

    def __init__(self, chunk=None):
        if chunk is None:
            chunk = np.tile(np.array([[1.0, -1.0, 0.5, -0.5, 1.0]], np.float32), (16, 1))
        self.chunk = np.asarray(chunk, dtype=np.float32)
        self.reset_calls = 0
        self.observe_calls = 0
        self.predict_calls = 0
        self.last_inference_seconds = 0.01

    def reset(self, observation, seed=1000):
        self.reset_calls += 1
        self.seed = seed

    def observe(self, observation):
        self.observe_calls += 1

    def predict_chunk(self):
        self.predict_calls += 1
        return sanitize_action_chunk(self.chunk)


class RL100DeploymentTest(unittest.TestCase):
    def test_observation_history_duplicates_reset_and_outputs_bcthw_contract(self):
        backend = MockBackend()
        first = validate_observation(backend._observation())
        history = MarvinObservationHistory(n_obs_steps=2)
        history.reset(first)
        batch = history.as_numpy()
        self.assertEqual(batch["agent_pos"].shape, (1, 2, 24))
        np.testing.assert_array_equal(batch["agent_pos"][0, 0], batch["agent_pos"][0, 1])
        self.assertEqual(set(batch).difference({"agent_pos"}), set(RL100_CAMERA_KEYS))
        for key in RL100_CAMERA_KEYS:
            self.assertEqual(batch[key].shape, (1, 2, 3, 128, 128))
            self.assertEqual(batch[key].dtype, np.uint8)

        backend.step_count = 1
        second = validate_observation(backend._observation())
        history.append(second)
        updated = history.as_numpy()
        self.assertNotEqual(float(updated["agent_pos"][0, 0, 9]), float(updated["agent_pos"][0, 1, 9]))

    def test_action_chunk_validation(self):
        chunk = np.array([[[1.2, -1.3, 0.0, 0.5, -1.0]]], np.float32)
        clipped, rows = sanitize_action_chunk(chunk)
        np.testing.assert_array_equal(clipped, np.array([[1.0, -1.0, 0.0, 0.5, -1.0]], np.float32))
        self.assertEqual(rows, 1)
        with self.assertRaises(SafetyError):
            sanitize_action_chunk(np.array([[np.nan] * 5], np.float32))
        with self.assertRaises(SafetyError):
            sanitize_action_chunk(np.zeros((2, 4), np.float32))

    def test_motion_scaling_does_not_change_gripper(self):
        result = scale_motion_action(np.array([1, -1, 0.5, -0.5, -1], np.float32), 0.2)
        np.testing.assert_allclose(result, [0.2, -0.2, 0.1, -0.1, -1.0])

    def test_chunk_queue_executes_two_actions_per_inference(self):
        client = FakeClient(success_at_step=3)
        policy = FakePolicy()
        contact = Mock()
        controller = RL100RolloutController(
            client=client,
            policy=policy,
            max_steps=10,
            actions_per_inference=2,
            motion_scale=0.2,
            bridge_contact_callback=contact,
        )
        result = controller.run_episode(episode_index=4)
        self.assertTrue(result["succeeded"])
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["inference_calls"], 2)
        self.assertEqual(policy.reset_calls, 1)
        self.assertEqual(policy.seed, 1004)
        self.assertEqual(policy.observe_calls, 3)
        self.assertEqual(len(client.actions), 3)
        self.assertEqual(contact.call_count, 4)  # reset plus three successful steps
        np.testing.assert_allclose(client.actions[0], [0.2, -0.2, 0.1, -0.1, 1.0])

    def test_primary_contact_clears_transient_heartbeat_failure(self):
        heartbeat = BridgeHeartbeat(Mock(), interval_s=0.5, max_consecutive_failures=3)
        heartbeat._consecutive_failures = 2
        heartbeat.last_error = RuntimeError("stale status timeout")
        heartbeat.note_primary_contact()
        heartbeat.check()
        self.assertEqual(heartbeat._consecutive_failures, 0)
        self.assertIsNone(heartbeat.last_error)

    def test_dry_run_never_calls_step(self):
        client = FakeClient(success_at_step=None)
        policy = FakePolicy()
        controller = RL100RolloutController(
            client=client,
            policy=policy,
            max_steps=3,
            actions_per_inference=1,
            motion_scale=0.2,
            dry_run=True,
        )
        result = controller.run_episode()
        self.assertTrue(result["dry_run"])
        self.assertEqual(len(client.actions), 0)
        self.assertEqual(policy.observe_calls, 0)
        self.assertEqual(policy.predict_calls, 3)

    def test_hardware_requires_explicit_opt_in(self):
        with self.assertRaises(RuntimeError):
            validate_bridge_preflight({"backend": "hardware"}, False)
        validate_bridge_preflight({"backend": "hardware"}, True)

    def test_replay_episode_bounds(self):
        ends = np.array([3, 8, 10], dtype=np.int64)
        self.assertEqual(episode_bounds(ends, 0), (0, 3))
        self.assertEqual(episode_bounds(ends, 1), (3, 8))
        self.assertEqual(episode_bounds(ends, 2), (8, 10))
        with self.assertRaises(IndexError):
            episode_bounds(ends, 3)

    def test_online_env_executes_chunk_and_aggregates_discounted_reward(self):
        client = FakeClient(success_at_step=3)
        env = MarvinBridgeOnlineEnv(
            client,
            n_obs_steps=2,
            n_action_steps=5,
            max_episode_steps=20,
        )
        obs, _ = env.reset()
        self.assertEqual(obs["agent_pos"].shape, (2, 24))
        chunk = np.zeros((5, 5), dtype=np.float32)
        next_obs, reward, done, info = env.step_online(chunk, gamma=0.9)
        self.assertTrue(done)
        self.assertTrue(info["terminated"])
        self.assertFalse(info["truncated"])
        self.assertEqual(info["executed_steps"], 3)
        self.assertAlmostEqual(reward, 0.9**2)
        self.assertEqual(next_obs["agent_pos"].shape, (2, 24))
        self.assertEqual(len(client.actions), 3)

    def test_online_env_rejects_action_modification(self):
        class ModifiedActionClient(FakeClient):
            def step(self, action):
                result = super().step(action)
                result["actual_action"] = result["actual_action"].copy()
                result["actual_action"][0] += 0.1
                return result

        client = ModifiedActionClient(success_at_step=None)
        env = MarvinBridgeOnlineEnv(client, n_action_steps=5)
        env.reset()
        with self.assertRaises(SafetyError):
            env.step_online(np.zeros((5, 5), dtype=np.float32))

    def test_online_env_refuses_to_clip_policy_action(self):
        client = FakeClient(success_at_step=None)
        env = MarvinBridgeOnlineEnv(client, n_action_steps=5)
        env.reset()
        chunk = np.zeros((5, 5), dtype=np.float32)
        chunk[0, 0] = 1.01
        with self.assertRaises(SafetyError):
            env.step_online(chunk)
        self.assertEqual(client.actions, [])

    def test_four_camera_online_buffer_round_trip(self):
        backend = MockBackend()
        history = MarvinObservationHistory(2)
        history.reset(validate_observation(backend._observation()))
        obs = {key: value[0] for key, value in history.as_numpy().items()}
        backend.step_count = 1
        history.append(validate_observation(backend._observation()))
        next_obs = {key: value[0] for key, value in history.as_numpy().items()}
        diffusion_path = np.zeros((11, 16, 5), dtype=np.float32)
        old_logprob = np.zeros((10, 16, 5), dtype=np.float32)

        buffer = MarvinOnlineReplayBuffer(capacity=2, device="cpu", policy_version=4)
        buffer.store(obs, diffusion_path, old_logprob, 0.0, next_obs, False, False)
        buffer.store(next_obs, diffusion_path, old_logprob, 1.0, next_obs, True, True)
        tensors = buffer.numpy_to_tensor()
        self.assertEqual(tensors[0]["agent_pos"].shape, (2, 2, 24))
        self.assertEqual(tensors[1].shape, (2, 11, 16, 5))
        self.assertEqual(tensors[2].shape, (2, 10, 16, 5))
        self.assertEqual(tensors[0]["rgb_side_policy"].dtype, tensors[1].dtype)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "buffer.pkl"
            buffer.save(path)
            restored = MarvinOnlineReplayBuffer(capacity=2, device="cpu", policy_version=4)
            restored.load(path, expected_policy_version=4)
            self.assertEqual(restored.count, 2)
            np.testing.assert_array_equal(restored.obs["rgb_wrist_1"], buffer.obs["rgb_wrist_1"])
            with self.assertRaises(ValueError):
                MarvinOnlineReplayBuffer(capacity=2, device="cpu").load(
                    path,
                    expected_policy_version=5,
                )


if __name__ == "__main__":
    unittest.main()
