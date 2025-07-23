import argparse
from collections import defaultdict
"""Analyze world model prediction error grouped by action."""

from pathlib import Path

import torch
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils import slice_trajdict_with_t
from plan import load_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze prediction error grouped by action class"
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to a trained model output directory containing hydra.yaml",
    )
    parser.add_argument(
        "--epoch",
        default="latest",
        help="Checkpoint epoch to load. Defaults to 'latest'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--round_decimals",
        type=int,
        default=2,
        help="Number of decimals to round actions for grouping (used if needed)",
    )
    return parser.parse_args()


def compute_error(z_pred, z_tgt, criterion) -> torch.Tensor:
    """Return mean embedding error per sample."""
    errors = {}
    for key in z_pred:
        errors[key] = criterion(z_pred[key], z_tgt[key]).view(-1)
    return torch.stack(list(errors.values()), dim=0).mean(dim=0)


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    cfg = OmegaConf.load(model_dir / "hydra.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets, traj_dsets = hydra.utils.call(
        cfg.env.dataset,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        frameskip=cfg.frameskip,
    )
    dset = datasets["valid"]
    traj_dset = traj_dsets["valid"]
    num_workers = int(getattr(cfg.env, "num_workers", 0))

    loader = DataLoader(
        dset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers
    )

    ckpt_dir = model_dir / "checkpoints"
    if args.epoch == "latest":
        ckpt_path = ckpt_dir / "model_latest.pth"
        if not ckpt_path.exists():
            candidates = list(ckpt_dir.glob("model_*.pth"))
            if not candidates:
                raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
            ckpt_path = max(candidates, key=lambda p: int(p.stem.split("_")[-1]))
    else:
        ckpt_path = ckpt_dir / f"model_{args.epoch}.pth"

    model = load_model(ckpt_path, cfg, cfg.num_action_repeat, device=device)
    model.eval()

    # Get normalization parameters
    action_mean = torch.tensor(traj_dset.action_mean.item(), device=device)
    action_std = torch.tensor(traj_dset.action_std.item(), device=device)


    error_by_action = defaultdict(list)

    with torch.no_grad():
        for obs, act, _ in loader:
            obs = {k: v.to(device) for k, v in obs.items()}
            act = act.to(device)

            z_out, _, _, _, _ = model(obs, act)
            z_obs_out, _ = model.separate_emb(z_out)

            z_gt = model.encode_obs(obs)
            z_tgt = slice_trajdict_with_t(z_gt, start_idx=model.num_pred)

            per_sample_err = compute_error(z_obs_out, z_tgt, model.emb_criterion)

            # De-normalize actions to extract true class label at timestep -1
            true_values = act[:, -1, 0] * action_std + action_mean
            class_labels = torch.floor(true_values + 0.5).long().cpu()


            for err, cls in zip(per_sample_err.cpu(), class_labels):
                error_by_action[int(cls.item())].append(err.item())

    print("Prediction error by action class:")
    for action, errs in sorted(error_by_action.items()):
        mean_err = sum(errs) / len(errs)
        print(f"Action {action}: {mean_err:.6f}")


if __name__ == "__main__":
    main()
