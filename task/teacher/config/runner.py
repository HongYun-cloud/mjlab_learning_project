"""Runner used by the Track tasks.

The upstream `PPO.load()` assumes that a checkpoint contains RND weights
whenever the current algorithm has an RND module.  Existing Track models were
trained without RND, so loading them would raise a ``KeyError``.  This runner
detects that case and starts the RND predictor from scratch while still loading
the actor / critic / optimizer.
"""

import os
from pathlib import Path

import torch
import wandb

from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class TrackOnPolicyRunner(VelocityOnPolicyRunner):
    def save(self, path: str, infos: dict | None = None) -> None:
        # Skip VelocityOnPolicyRunner.save() so ONNX is not exported for every
        # periodic checkpoint; only export the final policy after training.
        MjlabOnPolicyRunner.save(self, path, infos)

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        super().learn(num_learning_iterations, init_at_random_ep_len)
        self.export_final_onnx()

    def export_final_onnx(self) -> None:
        """Export the final teacher policy to the run directory."""
        log_dir = getattr(self.logger, "log_dir", None)
        if log_dir is None:
            return

        policy_dir = Path(log_dir)
        filename = f"{policy_dir.name}.onnx"
        onnx_path = policy_dir / filename
        try:
            self.export_policy_to_onnx(str(policy_dir), filename)
            run_name: str = (
                wandb.run.name
                if self.logger.logger_type in ("wandb", "WandbLogWriter")
                and wandb.run
                else policy_dir.name
            )
            metadata = get_base_metadata(self.env.unwrapped, run_name)
            attach_metadata_to_onnx(str(onnx_path), metadata)
        except Exception as e:
            print(f"[WARN] final ONNX export failed (training continues): {e}")
            return

        print(f"[ONNX] exported final teacher policy: {onnx_path}")
        if (
            self.logger.logger_type in ("wandb", "WandbLogWriter")
            and self.cfg["upload_model"]
        ):
            try:
                wandb.save(str(onnx_path), base_path=str(policy_dir))
            except Exception as e:
                print(f"[WARN] wandb ONNX upload skipped: {e}")

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        checkpoint = torch.load(
            path, map_location=map_location, weights_only=False
        )
        if "actor_state_dict" not in checkpoint:
            raise ValueError(
                f"Checkpoint '{path}' has no 'actor_state_dict'. "
                "Teacher/PPO training and playback require a teacher PPO "
                "checkpoint. Student distillation checkpoints store "
                "'student_state_dict' and cannot initialize a teacher. "
                f"Available keys: {sorted(checkpoint.keys())}."
            )
        if load_cfg is None:
            if "rnd_state_dict" not in checkpoint:
                print(
                    "[INFO] Checkpoint has no RND weights; loading actor/critic/"
                    "optimizer only and initializing RND from scratch."
                )
                load_cfg = {
                    "actor": True,
                    "critic": True,
                    "optimizer": True,
                    "iteration": True,
                    "rnd": False,
                }
        infos = super().load(path, load_cfg, strict, map_location)
        # Make step-based annealing start from this run, not the absolute
        # step count stored in the resumed checkpoint.
        self.env.unwrapped._anneal_start_step = (
            self.env.unwrapped.common_step_counter
        )
        return infos
