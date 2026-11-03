"""Runner used by the Track tasks.

The upstream `PPO.load()` assumes that a checkpoint contains RND weights
whenever the current algorithm has an RND module.  Existing Track models were
trained without RND, so loading them would raise a ``KeyError``.  This runner
detects that case and starts the RND predictor from scratch while still loading
the actor / critic / optimizer.
"""

import torch

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class TrackOnPolicyRunner(VelocityOnPolicyRunner):
    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        if load_cfg is None:
            checkpoint = torch.load(
                path, map_location=map_location, weights_only=False
            )
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
        return super().load(path, load_cfg, strict, map_location)
