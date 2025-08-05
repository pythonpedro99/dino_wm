import argparse
from collections import defaultdict
"""Analyze world model prediction error grouped by action."""
import torch.nn as nn

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
    """Return mean embedding error per sample (along batch)."""
    per_sample_errors = []
    for key in z_pred:
        # criterion returns (B, ...) → reduce only non-batch dims
        err = criterion(z_pred[key], z_tgt[key])
        while err.ndim > 1:
            err = err.mean(dim=-1)  # reduce feature dims
        per_sample_errors.append(err)  # shape (B,)
    
    # Stack all keys (e.g., visual, proprio...) → shape (num_keys, B)
    per_sample_errors = torch.stack(per_sample_errors, dim=0)

    # Average over keys → shape (B,)
    return per_sample_errors.mean(dim=0)



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
    print(f"Total number of samples in dataset: {len(dset)}")
    print(f"Total number of batches: {len(loader)}")


    with torch.no_grad():
        for obs, act, _ in loader:
            obs = {k: v.to(device) for k, v in obs.items()}
            act = act.to(device)
            print(f"Processing batch with size: {act.shape}")


            z_out, _, _, _, _ = model(obs, act)
            z_obs_out, _ = model.separate_emb(z_out)

            z_gt = model.encode_obs(obs)
            z_tgt = slice_trajdict_with_t(z_gt, start_idx=model.num_pred)
            print(model.emb_criterion)

            per_sample_err = compute_error(z_obs_out, z_tgt, nn.MSELoss(reduction='none'))
            print(f"per_sample_err shape: {per_sample_err.shape}")



            # De-normalize actions to extract true class label at timestep -1
            true_values = act[:, -1, 0] * action_std + action_mean
            class_labels = torch.floor(true_values + 0.5).long().cpu()
            print(f"class_labels shape: {class_labels.shape}")
            #print(f"true_value:{true_values}, classlabel:{class_labels}")
            for i, (err, cls) in enumerate(zip(per_sample_err.cpu(), class_labels)):
              print(f"[{i}] class={cls.item()}, err={err.item()}")
              error_by_action[int(cls.item())].append(err.item())


    print("Prediction error by action class:")
    for action, errs in sorted(error_by_action.items()):
        mean_err = sum(errs) / len(errs)
        print(f"Action {action}: {mean_err:.6f}")


if __name__ == "__main__":
    main()