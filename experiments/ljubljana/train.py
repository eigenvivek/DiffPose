from itertools import product
from pathlib import Path

import submitit
import torch
from diffdrr.drr import DRR
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from pytorch_transformers.optimization import WarmupCosineSchedule
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from tqdm import tqdm

from diffpose.ljubljana import LjubljanaDataset, Transforms, get_random_offset
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from diffpose.registration import PoseRegressor


def load(id_number, view, subsample, device):
    # Load the subject
    subject = LjubljanaDataset(view)
    (
        volume,
        spacing,
        focal_len,
        height,
        width,
        delx,
        dely,
        x0,
        y0,
        _,
        _,
        isocenter_pose,
    ) = subject[id_number]
    volume[volume < 500] = 0.0
    isocenter_pose = isocenter_pose.to(device)

    # Make the DRR
    height //= subsample
    width //= subsample
    delx *= subsample
    dely *= subsample

    drr = DRR(
        volume,
        spacing,
        focal_len / 2,
        height,
        delx,
        width,
        dely,
        x0,
        y0,
        reverse_x_axis=True,
    ).to(device)
    transforms = Transforms(height, width)

    return drr, isocenter_pose, transforms


def train(
    model,
    optimizer,
    scheduler,
    drr,
    transforms,
    isocenter_pose,
    batch_size,
    n_epochs,
    n_batches_per_epoch,
    model_params,
    id_number,
    view,
    device,
):
    metric = MultiscaleNormalizedCrossCorrelation2d(eps=1e-4)
    geodesic = GeodesicSE3()
    double = DoubleGeodesic(drr.detector.sdr)

    best_loss = torch.inf

    model.train()
    for epoch in range(n_epochs):
        losses = []
        for _ in (itr := tqdm(range(n_batches_per_epoch), leave=False)):
            try:
                offset = get_random_offset(view, batch_size, device)
                pose = isocenter_pose.compose(offset)
                img = drr(None, None, None, pose=pose)
                img = transforms(img)
    
                pred_offset = model(img)
                pred_pose = isocenter_pose.compose(pred_offset)
                pred_img = drr(None, None, None, pose=pred_pose)
                pred_img = transforms(pred_img)
    
                ncc = metric(pred_img, img)
                log_geodesic = geodesic(pred_pose, pose)
                geodesic_rot, geodesic_xyz, double_geodesic = double(pred_pose, pose)
                loss = 1 - ncc + 1e-2 * (log_geodesic + double_geodesic)
    
                optimizer.zero_grad()
                loss.mean().backward()
                adaptive_clip_grad_(model.parameters())
                optimizer.step()
                scheduler.step()
    
                losses.append(loss.mean().item())
    
                # Update progress bar
                itr.set_description(f"Epoch [{epoch}/{n_epochs}]")
                itr.set_postfix(
                    geodesic_rot=geodesic_rot.mean().item(),
                    geodesic_xyz=geodesic_xyz.mean().item(),
                    geodesic_dou=double_geodesic.mean().item(),
                    geodesic_se3=log_geodesic.mean().item(),
                    loss=loss.mean().item(),
                    ncc=ncc.mean().item(),
                )
    
                prev_pose = pose
                prev_pred_pose = pred_pose
            except:
                print("Aaaaaaand we've crashed...")
                print(ncc)
                print(log_geodesic)
                print(geodesic_rot)
                print(geodesic_xyz)
                print(double_geodesic)
                print(pose.get_matrix())
                print(pred_pose.get_matrix())
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "height": drr.detector.height,
                        "width": drr.detector.width,
                        "epoch": epoch,
                        "batch_size": batch_size,
                        "n_epochs": n_epochs,
                        "n_batches_per_epoch": n_batches_per_epoch,
                        "pose": pose.get_matrix().cpu(),
                        "pred_pose": pred_pose.get_matrix().cpu(),
                        **model_params,
                    },
                    f"checkpoints/specimen_{id_number:02d}_{view}_crashed.ckpt",
                )
                raise RuntimeError("NaN loss")

        losses = torch.tensor(losses)
        tqdm.write(f"Epoch {epoch + 1:04d} | Loss {losses.mean().item():.4f}")
        if losses.mean() < best_loss and not losses.isnan().any():
            best_loss = losses.mean().item()
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "height": drr.detector.height,
                    "width": drr.detector.width,
                    "epoch": epoch,
                    "loss": losses.mean().item(),
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "n_batches_per_epoch": n_batches_per_epoch,
                    **model_params,
                },
                f"checkpoints/specimen_{id_number:02d}_{view}_best.ckpt",
            )

        if epoch % 25 == 0 and epoch != 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "height": drr.detector.height,
                    "width": drr.detector.width,
                    "epoch": epoch,
                    "loss": losses.mean().item(),
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "n_batches_per_epoch": n_batches_per_epoch,
                    **model_params,
                },
                f"checkpoints/specimen_{id_number:02d}_{view}_epoch{epoch:03d}.ckpt",
            )

def main(
    id_number,
    view,
    subsample=8,
    restart=None,
    model_name="resnet18",
    parameterization="se3_log_map",
    convention=None,
    lr=1e-3,
    batch_size=1,
    n_epochs=1000,
    n_batches_per_epoch=100,
):
    device = torch.device("cuda")
    drr, isocenter_pose, transforms = load(id_number, view, subsample, device)

    model_params = {
        "model_name": model_name,
        "parameterization": parameterization,
        "convention": convention,
        "norm_layer": "groupnorm",
    }
    model = PoseRegressor(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if restart is not None:
        ckpt = torch.load(restart)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    model = model.to(device)

    scheduler = WarmupCosineSchedule(
        optimizer,
        5 * n_batches_per_epoch,
        n_epochs * n_batches_per_epoch - 5 * n_batches_per_epoch,
    )

    train(
        model,
        optimizer,
        scheduler,
        drr,
        transforms,
        isocenter_pose,
        batch_size,
        n_epochs,
        n_batches_per_epoch,
        model_params,
        id_number,
        view,
        device,
    )

if __name__ == "__main__":
    id_numbers = list(range(10))
    views = ["ap", "lat"]
    id_numbers = [i for i, _ in product(id_numbers, views)]
    views = [v for _, v in product(id_numbers, views)]
    
    Path("checkpoints").mkdir(exist_ok=True)

    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="ljubljana",
        gpus_per_node=1,
        mem_gb=10.0,
        slurm_array_parallelism=len(id_numbers),
        slurm_partition="2080ti",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, id_numbers, views)
