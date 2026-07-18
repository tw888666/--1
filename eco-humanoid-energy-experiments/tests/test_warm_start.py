import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from bruce_gym.algo.rl.on_policy_runner import OnPolicyRunner


class _TinyActorCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(2, 2)
        self.critic = torch.nn.Linear(2, 1)
        self.cost_critic = torch.nn.Linear(2, 1)
        self.std = torch.nn.Parameter(torch.ones(2))


class _TinyLagrange:
    def __init__(self, value):
        self.lagrangian_multiplier = torch.nn.Parameter(torch.tensor(value))
        self.lambda_optimizer = torch.optim.Adam([self.lagrangian_multiplier], lr=1e-3)
        self.previous_cost = 123.0


def _fill(module, value):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)


def _make_runner(log_dir):
    actor_critic = _TinyActorCritic()
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
    cost_optimizer = torch.optim.Adam(actor_critic.cost_critic.parameters(), lr=1e-3)
    runner = object.__new__(OnPolicyRunner)
    runner.device = "cpu"
    runner.alg = SimpleNamespace(
        actor_critic=actor_critic,
        optimizer=optimizer,
        cost_value_optimizer=cost_optimizer,
    )
    runner.alg_cfg = {
        "lagrangian_multiplier_init1": 0.0,
        "lagrangian_multiplier_init2": 0.0,
    }
    runner.constraint_num = 2
    runner.lagrange1 = _TinyLagrange(5.0)
    runner.lagrange2 = _TinyLagrange(6.0)
    runner.current_learning_iteration = 99
    runner.tot_timesteps = 100
    runner.tot_time = 200.0
    runner.smooth_cost_mean = 3.0
    runner.last_update = 4
    runner.log_dir = str(log_dir)
    runner.env = SimpleNamespace(
        num_envs=8192,
        energy_cost_mode="joint_abs_8",
        cfg=SimpleNamespace(env=SimpleNamespace(cost_limit1=56.274914)),
    )
    runner.all_cfg = {"seed": 123}
    runner.initialization_metadata = {"mode": "fresh"}
    return runner


class WarmStartTest(unittest.TestCase):
    def test_warm_start_loads_policy_and_resets_cost_training_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = _TinyActorCritic()
            _fill(source.actor, 1.0)
            _fill(source.critic, 2.0)
            _fill(source.cost_critic, 9.0)
            with torch.no_grad():
                source.std.fill_(3.0)
            checkpoint = temp_path / "model_3000.pt"
            torch.save(
                {
                    "model_state_dict": source.state_dict(),
                    "iter": 3000,
                    "infos": None,
                },
                checkpoint,
            )

            runner = _make_runner(temp_path / "run")
            metadata = runner.load_warm_start(str(checkpoint), reset_seed=123)

            self.assertTrue(torch.equal(runner.alg.actor_critic.actor.weight, source.actor.weight))
            self.assertTrue(torch.equal(runner.alg.actor_critic.critic.weight, source.critic.weight))
            self.assertTrue(torch.equal(runner.alg.actor_critic.std, source.std))
            self.assertFalse(
                torch.equal(
                    runner.alg.actor_critic.cost_critic.weight,
                    source.cost_critic.weight,
                )
            )
            self.assertEqual(runner.alg.optimizer.state, {})
            self.assertEqual(runner.alg.cost_value_optimizer.state, {})
            self.assertEqual(runner.lagrange1.lagrangian_multiplier.item(), 0.0)
            self.assertEqual(runner.lagrange2.lagrangian_multiplier.item(), 0.0)
            self.assertIsNone(runner.lagrange1.previous_cost)
            self.assertEqual(runner.current_learning_iteration, 0)
            self.assertEqual(runner.tot_timesteps, 0)
            self.assertEqual(runner.tot_time, 0)
            self.assertEqual(metadata["source_checkpoint_iteration"], 3000)
            self.assertEqual(metadata["energy_cost_mode"], "joint_abs_8")

            metadata_path = temp_path / "run" / "warm_start_metadata.json"
            self.assertTrue(metadata_path.is_file())
            written = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(written["source_checkpoint_sha256"], metadata["source_checkpoint_sha256"])

    def test_cost_critic_reset_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = _TinyActorCritic()
            checkpoint = temp_path / "model_3000.pt"
            torch.save(
                {"model_state_dict": source.state_dict(), "iter": 3000, "infos": None},
                checkpoint,
            )

            first = _make_runner(temp_path / "first")
            second = _make_runner(temp_path / "second")
            _fill(first.alg.actor_critic.cost_critic, 7.0)
            _fill(second.alg.actor_critic.cost_critic, 8.0)
            first_metadata = first.load_warm_start(str(checkpoint), reset_seed=123)
            second_metadata = second.load_warm_start(str(checkpoint), reset_seed=123)

            self.assertEqual(
                first_metadata["cost_critic_initial_sha256"],
                second_metadata["cost_critic_initial_sha256"],
            )
            for name, tensor in first.alg.actor_critic.cost_critic.state_dict().items():
                self.assertTrue(
                    torch.equal(tensor, second.alg.actor_critic.cost_critic.state_dict()[name])
                )


if __name__ == "__main__":
    unittest.main()
