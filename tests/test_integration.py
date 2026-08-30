import tempfile
import threading
import time
import unittest
from socketserver import ThreadingMixIn
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lerobot_robot_marvin.backends import HilserlBackend, HumanOutcomeListener, MockBackend
from lerobot_robot_marvin.act_rollout import sanitize_policy_action, validate_bridge_preflight
from lerobot_robot_marvin.bridge_server import BridgeHTTPServer, BridgeService, make_backend
from lerobot_robot_marvin.client import MarvinBridgeClient
from lerobot_robot_marvin.config_marvin import MarvinRobotConfig
from lerobot_robot_marvin.constants import CAMERA_KEYS, ROBOT_ACTION_KEYS, ROBOT_STATE_KEYS
from lerobot_robot_marvin.episode import MarvinEpisodeController
from lerobot_robot_marvin.errors import BridgeError, SafetyError
from lerobot_robot_marvin.marvin import MarvinRobot
from lerobot_robot_marvin.schema import validate_action, validate_observation


class InProcessClient:
    """Exercises the real BridgeService without opening a sandbox-forbidden socket."""

    def __init__(self, service, fail_step=False):
        self.service = service
        self.fail_step = fail_step
        self.connected = False

    def connect(self):
        result = self.service.dispatch("connect", {})
        self.connected = True
        return result

    def reset(self):
        result = self.service.dispatch("reset", {})
        return validate_observation(result["observation"]), result["info"]

    def observe(self):
        return validate_observation(self.service.dispatch("observe", {})["observation"])

    def step(self, action):
        if self.fail_step:
            raise BridgeError("simulated transport timeout")
        result = self.service.dispatch("step", {"action": validate_action(action)})
        result["observation"] = validate_observation(result["observation"])
        result["actual_action"] = validate_action(result["actual_action"])
        return result

    def stop(self, reason="test stop"):
        return self.service.dispatch("stop", {"reason": reason})

    def best_effort_stop(self, reason):
        try:
            self.stop(reason)
        except Exception:
            pass

    def recover(self):
        return self.service.dispatch("recover", {})

    def status(self):
        return self.service.dispatch("status", {})

    def disconnect(self):
        if self.connected:
            self.service.dispatch("disconnect", {})
            self.connected = False


class Fixture:
    def __init__(self, backend, watchdog_timeout_s=5.0, fail_step=False):
        self.backend = backend
        self.service = BridgeService(backend, watchdog_timeout_s)
        self.client = InProcessClient(self.service, fail_step=fail_step)

    def close(self):
        self.service.close()


class FakePolicy:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class FakeOutcomeListener:
    def __init__(self, outcome=None):
        self.outcome = outcome
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def consume(self):
        outcome, self.outcome = self.outcome, None
        return outcome


class FakeHilserlEnv:
    def __init__(self):
        self.step_calls = 0

    def reset(self):
        return MockBackend()._observation(), {"succeed": False}

    def step(self, action):
        self.step_calls += 1
        return MockBackend()._observation(), 0.0, False, False, {"succeed": False}


class SlowStopBackend(MockBackend):
    def __init__(self, stop_delay_s):
        super().__init__()
        self.stop_delay_s = stop_delay_s

    def stop(self, reason):
        time.sleep(self.stop_delay_s)
        super().stop(reason)


class AppliedActionBackend(MockBackend):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["intervene_action"] = np.array([0.9, 0.0, 0.0, 0.0, 0.0], np.float32)
        info["applied_command_action"] = np.array([0.8, 0.0, 0.0, 0.0, 0.0], np.float32)
        return observation, reward, terminated, truncated, info


class FakeInputProcess:
    def __init__(self, events):
        self.events = events

    def join(self, timeout=None):
        self.events.append(("process.join", timeout))


class FakeInputManager:
    def __init__(self, events):
        self.events = events

    def shutdown(self):
        self.events.append("manager.shutdown")


class FakeInputExpert:
    def __init__(self, events):
        self.events = events
        self.process = FakeInputProcess(events)
        self.manager = FakeInputManager(events)

    def close(self):
        self.events.append("expert.close")


class FakeWrappedHardwareEnv:
    def __init__(self, events):
        self.events = events
        self.expert = FakeInputExpert(events)

    def close(self):
        self.events.append("env.close")


def zero_action():
    return dict.fromkeys(ROBOT_ACTION_KEYS, 0.0)


class IntegrationTest(unittest.TestCase):
    def make_robot(self, fixture):
        calibration = Path(tempfile.mkdtemp(prefix="marvin-calibration-"))
        config = MarvinRobotConfig(id="test", calibration_dir=calibration)
        return MarvinRobot(config, client=fixture.client)

    def test_schema_action_order_and_next_observation_cache(self):
        fixture = Fixture(MockBackend())
        self.addCleanup(fixture.close)
        robot = self.make_robot(fixture)
        robot.connect()
        robot.reset_episode()
        obs = robot.get_observation()
        self.assertEqual(tuple(key for key in obs if key in ROBOT_STATE_KEYS), ROBOT_STATE_KEYS)
        self.assertEqual({key: obs[key].shape for key in CAMERA_KEYS}, dict.fromkeys(CAMERA_KEYS, (128, 128, 3)))
        self.assertEqual(fixture.backend.observe_calls, 0)

        action = {key: float(index) / 10 for index, key in enumerate(ROBOT_ACTION_KEYS)}
        robot.send_action(action)
        np.testing.assert_allclose(fixture.backend.last_action, np.arange(5) / 10)
        next_obs = robot.get_observation()
        self.assertAlmostEqual(next_obs[ROBOT_STATE_KEYS[1]], 0.0)
        self.assertAlmostEqual(next_obs[ROBOT_STATE_KEYS[5]], 0.4)
        self.assertEqual(fixture.backend.observe_calls, 0)
        robot.disconnect()
        self.assertGreaterEqual(fixture.backend.stop_calls, 1)
        self.assertEqual(fixture.backend.disconnect_calls, 1)

    def test_bridge_returns_safety_clipped_pre_smoothing_action(self):
        fixture = Fixture(AppliedActionBackend())
        self.addCleanup(fixture.close)
        fixture.client.connect()
        fixture.client.reset()
        transition = fixture.client.step(np.array([0.2, 0.0, 0.0, 0.0, 0.0], np.float32))
        np.testing.assert_allclose(transition["requested_action"], [0.9, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(transition["applied_command_action"], [0.8, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(transition["actual_action"], transition["applied_command_action"])
        self.assertTrue(transition["intervened"])

    def test_invalid_actions_fail_safe_stop(self):
        for invalid in (float("nan"), 1.1):
            with self.subTest(invalid=invalid):
                fixture = Fixture(MockBackend())
                robot = self.make_robot(fixture)
                robot.connect()
                robot.reset_episode()
                action = zero_action()
                action[ROBOT_ACTION_KEYS[0]] = invalid
                with self.assertRaises(SafetyError):
                    robot.send_action(action)
                self.assertGreaterEqual(fixture.backend.stop_calls, 1)
                robot.disconnect()
                fixture.close()

    def test_transport_failure_triggers_stop(self):
        fixture = Fixture(MockBackend(), fail_step=True)
        self.addCleanup(fixture.close)
        robot = self.make_robot(fixture)
        robot.connect()
        robot.reset_episode()
        with self.assertRaises(BridgeError):
            robot.send_action(zero_action())
        self.assertGreaterEqual(fixture.backend.stop_calls, 1)
        robot.disconnect()

    def test_episode_reset_success_and_policy_queue_reset(self):
        fixture = Fixture(MockBackend(success_at_step=3))
        self.addCleanup(fixture.close)
        robot = self.make_robot(fixture)
        robot.connect()
        policy = FakePolicy()
        result = MarvinEpisodeController(robot, max_steps=10).run_episode(policy, lambda obs: zero_action())
        self.assertTrue(result.succeeded)
        self.assertEqual(result.steps, 3)
        self.assertEqual(policy.reset_calls, 1)
        self.assertEqual(fixture.backend.reset_calls, 1)
        robot.disconnect()

    def test_watchdog_stops_disconnected_client(self):
        fixture = Fixture(MockBackend(), watchdog_timeout_s=0.05)
        self.addCleanup(fixture.close)
        robot = self.make_robot(fixture)
        robot.connect()
        deadline = time.monotonic() + 0.5
        while fixture.backend.stop_calls == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertGreaterEqual(fixture.backend.stop_calls, 1)
        robot.disconnect()

    def test_explicit_slow_stop_does_not_reenter_watchdog_stop(self):
        backend = SlowStopBackend(stop_delay_s=0.08)
        fixture = Fixture(backend, watchdog_timeout_s=0.03)
        self.addCleanup(fixture.close)
        fixture.client.connect()
        fixture.client.stop("test terminal cleanup")
        self.assertEqual(backend.stop_calls, 1)
        self.assertEqual(backend.last_stop_reason, "test terminal cleanup")

    def test_hardware_stop_closes_input_worker_before_environment(self):
        events = []
        backend = HilserlBackend()
        backend.env = FakeWrappedHardwareEnv(events)
        backend.stop("test orderly shutdown")
        self.assertEqual(
            events,
            ["expert.close", ("process.join", 1.0), "manager.shutdown", "env.close"],
        )

    def test_long_connect_and_reset_timeouts_are_used(self):
        client = MarvinBridgeClient(connect_timeout_s=31.0, reset_timeout_s=121.0)
        observation = MockBackend()._observation()
        with patch.object(
            client,
            "_rpc",
            side_effect=[
                {"backend": "mock"},
                {"observation": observation, "info": {}},
                {"disconnected": True},
            ],
        ) as rpc:
            client.connect()
            client.reset()
            client.disconnect()
        self.assertEqual(rpc.call_args_list[0].kwargs["timeout_s"], 31.0)
        self.assertEqual(rpc.call_args_list[1].kwargs["timeout_s"], 121.0)
        self.assertEqual(rpc.call_args_list[2].kwargs["timeout_s"], 31.0)

    def test_connect_transport_failure_requests_fail_safe_stop(self):
        client = MarvinBridgeClient()
        with patch.object(client, "_rpc", side_effect=BridgeError("lost response")), patch.object(
            client, "best_effort_stop"
        ) as stop:
            with self.assertRaises(BridgeError):
                client.connect()
        stop.assert_called_once_with("connect RPC failed or timed out")
        self.assertFalse(client.connected)

    def test_hardware_preflight_and_non_loopback_bind_are_refused(self):
        with self.assertRaises(RuntimeError):
            validate_bridge_preflight({"backend": "hardware"}, allow_real_robot=False)
        validate_bridge_preflight({"backend": "hardware"}, allow_real_robot=True)
        args = SimpleNamespace(
            backend="hardware",
            host="0.0.0.0",
            allow_hardware=True,
            confirm_hardware_risk="I_UNDERSTAND_THIS_CONTROLS_A_REAL_ROBOT",
        )
        with self.assertRaises(SystemExit):
            make_backend(args)

    def test_bridge_requests_run_without_threading_mixin(self):
        # Hardware environment construction installs signal handlers, which
        # Python only allows in the main interpreter thread.
        self.assertFalse(issubclass(BridgeHTTPServer, ThreadingMixIn))

    def test_client_adopts_preconnected_backend_without_connecting_twice(self):
        backend = MockBackend()
        backend.connect()
        fixture = Fixture(backend)
        self.addCleanup(fixture.close)
        with patch.object(backend, "connect", wraps=backend.connect) as connect:
            status = fixture.client.connect()
        connect.assert_not_called()
        self.assertTrue(status["connected"])

    def test_service_close_releases_preconnected_backend_before_client(self):
        backend = MockBackend()
        backend.connect()
        service = BridgeService(backend)
        service.close()
        self.assertFalse(backend.connected)
        self.assertEqual(backend.disconnect_calls, 1)

    def test_policy_output_is_clipped_but_nonfinite_is_rejected(self):
        clipped, changed = sanitize_policy_action(np.array([1.05, -1.2, 0.0, 0.5, -0.5], np.float32))
        np.testing.assert_array_equal(clipped, np.array([1.0, -1.0, 0.0, 0.5, -0.5], np.float32))
        self.assertTrue(changed)
        with self.assertRaises(SafetyError):
            sanitize_policy_action(np.array([np.nan, 0.0, 0.0, 0.0, 0.0], np.float32))

    def test_human_success_key_overrides_transition_without_classifier(self):
        backend = HilserlBackend(classifier=False, human_success_key=True)
        backend.env = FakeHilserlEnv()
        backend._last_observation = validate_observation(MockBackend()._observation())
        backend._outcome_listener = FakeOutcomeListener("success")
        observation, reward, terminated, truncated, info = backend.step(np.zeros(5, np.float32))
        self.assertEqual(observation["state"].shape, (24,))
        self.assertEqual(reward, 1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["succeed"])
        self.assertEqual(info["human_outcome"], "success")
        self.assertEqual(backend.env.step_calls, 0)

    def test_human_escape_key_marks_terminal_failure(self):
        backend = HilserlBackend(classifier=False, human_success_key=True)
        backend.env = FakeHilserlEnv()
        backend._last_observation = validate_observation(MockBackend()._observation())
        backend._outcome_listener = FakeOutcomeListener("failure")
        _, reward, terminated, truncated, info = backend.step(np.zeros(5, np.float32))
        self.assertEqual(reward, -0.5)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["succeed"])
        self.assertEqual(info["failure_reason"], "keyboard_esc")
        self.assertEqual(backend.env.step_calls, 0)

    def test_human_footswitch_failure_is_identified(self):
        backend = HilserlBackend(
            classifier=False,
            human_success_key=True,
            human_outcome_input="footswitch",
        )
        backend.env = FakeHilserlEnv()
        backend._last_observation = validate_observation(MockBackend()._observation())
        backend._outcome_listener = FakeOutcomeListener("failure")
        _, _, terminated, _, info = backend.step(np.zeros(5, np.float32))
        self.assertTrue(terminated)
        self.assertEqual(info["failure_reason"], "footswitch_shift_right")

    def test_human_outcome_listener_switches_input_mode(self):
        keys = SimpleNamespace(
            shift="shift",
            shift_l="shift_l",
            shift_r="shift_r",
            left="left",
            right="right",
            space="space",
            esc="esc",
        )
        def make_listener(input_mode):
            listener = HumanOutcomeListener.__new__(HumanOutcomeListener)
            listener._keyboard = SimpleNamespace(Key=keys)
            listener._input_mode = input_mode
            listener._lock = threading.Lock()
            listener._success = False
            listener._failure = False
            listener._pressed_shifts = set()
            return listener

        listener = make_listener("footswitch")
        listener._on_press(keys.shift_l)
        listener._on_press(keys.left)
        self.assertEqual(listener.consume(), "success")
        listener._on_release(keys.shift_l)
        listener._on_press(keys.left)
        self.assertIsNone(listener.consume())

        listener._on_press(keys.shift_r)
        listener._on_press(keys.right)
        self.assertEqual(listener.consume(), "failure")
        listener._on_press(keys.space)
        listener._on_press(keys.esc)
        self.assertIsNone(listener.consume())

        listener = make_listener("keyboard")
        listener._on_press(keys.space)
        self.assertEqual(listener.consume(), "success")
        listener._on_press(keys.esc)
        self.assertEqual(listener.consume(), "failure")
        listener._on_press(keys.shift_l)
        listener._on_press(keys.left)
        self.assertIsNone(listener.consume())


if __name__ == "__main__":
    unittest.main()
