import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from act_rl.algorithms.iql import IQLACTLearner, expectile_loss
from act_rl.config import ACTRLConfig, IQLConfig, NetworkConfig, OnlineConfig
from act_rl.datasets.lerobot_chunk import LeRobotChunkDataset
from act_rl.replay import MixedReplay, ReplayBuffer
from act_rl.types import ChunkBatch, ChunkTransition
from act_rl.scripts.train_online import resolve_offline_fraction


class FakeActor(nn.Module):
    def __init__(self, chunk=3, action=2):
        super().__init__()
        self.net = nn.Linear(4, chunk * action)

    def parameter_groups(self, actor_lr, backbone_lr):
        return [{"params": self.parameters(), "lr": actor_lr}]

    def weighted_loss(self, batch, weights, prior_weight=0):
        pred = self.net(batch.state).reshape_as(batch.action)
        per = (pred - batch.action).abs().mean((1, 2))
        loss = (per * weights).sum() / weights.sum().clamp_min(1e-6)
        return loss, {"weighted_l1": per.mean().item(), "kld": 0.0, "prior": 0.0}


def transition(source="offline"):
    return ChunkTransition(
        state=np.zeros(4, np.float32), action_chunk=np.zeros((3, 2), np.float32),
        next_state=np.ones(4, np.float32), reward=1.0, discount=0.9,
        terminated=True, action_is_pad=np.zeros(3, bool), source=source,
    )


class ACTRLCoreTest(unittest.TestCase):
    def test_transition_validation(self):
        transition().validate(4, 3, 2)

    def test_mixed_replay_falls_back_to_offline(self):
        offline, online = ReplayBuffer(seed=1), ReplayBuffer(seed=2)
        offline.add(transition())
        batch = MixedReplay(offline, online).sample(4)
        self.assertEqual(tuple(batch.action.shape), (4, 3, 2))

    def test_expectile_is_asymmetric(self):
        value = expectile_loss(torch.tensor([-1.0, 1.0]), 0.7)
        self.assertAlmostEqual(value.item(), 0.5, places=5)

    def test_stage1_offline_fraction_schedule(self):
        cfg = ACTRLConfig(online=OnlineConfig())
        self.assertEqual(resolve_offline_fraction(cfg, 999), 0.75)
        self.assertEqual(resolve_offline_fraction(cfg, 1000), 0.50)
        self.assertEqual(resolve_offline_fraction(cfg, 4999), 0.50)
        self.assertEqual(resolve_offline_fraction(cfg, 5000), 0.25)

    def test_iql_update(self):
        cfg = ACTRLConfig(
            network=NetworkConfig(state_dim=4, action_dim=2, chunk_size=3, executed_steps=1,
                                  critic_horizon=1, hidden_dims=(16, 16), use_images=False),
            iql=IQLConfig(actor_lr=1e-3, critic_lr=1e-3, value_lr=1e-3), batch_size=4,
        )
        learner = IQLACTLearner(FakeActor(), cfg)
        batch = ReplayBuffer(seed=0)
        batch.extend([transition() for _ in range(8)])
        metrics = learner.update(batch.sample(4))
        self.assertIn("actor_loss", metrics)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))

    def test_recorded_n_step_reward_and_terminal_boundary(self):
        class FakeDataset:
            meta = SimpleNamespace(episodes=[{"episode_index": 0, "length": 3}])

            def __len__(self):
                return 3

            def __getitem__(self, index):
                # Mimic LeRobot delta windows for h=2.
                rewards = ([1.0, 2.0], [2.0, 3.0], [3.0, 3.0])[index]
                terminals = ([False, False], [False, True], [True, True])[index]
                return {
                    "episode_index": torch.tensor(0), "frame_index": torch.tensor(index),
                    "observation.state": torch.zeros(2, 4), "action": torch.zeros(3, 2),
                    "action_is_pad": torch.zeros(3, dtype=torch.bool),
                    "rl.reward": torch.tensor(rewards).unsqueeze(-1),
                    "rl.terminated": torch.tensor(terminals).unsqueeze(-1),
                }

        dataset = LeRobotChunkDataset(
            FakeDataset(), image_keys=(), executed_steps=2, gamma=0.5,
            outcomes={0: {"success": True, "source": "test"}},
        )
        first, second, last = dataset[0], dataset[1], dataset[2]
        self.assertAlmostEqual(float(first["reward"]), 2.0)  # 1 + .5 * 2
        self.assertFalse(bool(first["terminated"]))
        self.assertAlmostEqual(float(second["reward"]), 3.5)  # 2 + .5 * 3
        self.assertTrue(bool(second["terminated"]))
        self.assertAlmostEqual(float(last["reward"]), 3.0)
        self.assertAlmostEqual(float(last["discount"]), 0.5)


if __name__ == "__main__":
    unittest.main()
