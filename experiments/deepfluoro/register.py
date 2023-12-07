import time
from itertools import product
from pathlib import Path

import pandas as pd
import submitit
import torch
from diffdrr.drr import DRR
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from torchvision.transforms.functional import resize
from tqdm import tqdm

from diffpose.calibration import RigidTransform, convert
from diffpose.deepfluoro import DeepFluoroDataset, Evaluator, Transforms
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from diffpose.registration import PoseRegressor, SparseRegistration


class Registration:
    def __init__(
        self,
        drr,
        specimen,
        model,
        parameterization,
        convention=None,
        n_iters=500,
        verbose=False,
        device="cuda",
    ):
        self.device = torch.device(device)
        self.drr = drr.to(self.device)
        self.model = model.to(self.device)
        model.eval()

        self.specimen = specimen
        self.isocenter_pose = specimen.isocenter_pose.to(self.device)

        self.geodesics = GeodesicSE3()
        self.doublegeo = DoubleGeodesic(sdr=self.specimen.focal_len / 2)
        self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 13], [0.5, 0.5])
        self.transforms = Transforms(self.drr.detector.height)
        self.parameterization = parameterization
        self.convention = convention

        self.n_iters = n_iters
        self.verbose = verbose

    def initialize_registration(self, img):
        with torch.no_grad():
            offset = self.model(img)
            features = self.model.backbone.forward_features(img)
            features = resize(
                features,
                (self.drr.detector.height, self.drr.detector.width),
                interpolation=3,
                antialias=True,
            )
            features = features.sum(dim=[0, 1], keepdim=True)
            features -= features.min()
            features /= features.max() - features.min()
            features /= features.sum()
        pred_pose = self.isocenter_pose.compose(offset)

        return SparseRegistration(
            self.drr,
            pose=pred_pose,
            parameterization=self.parameterization,
            convention=self.convention,
            features=features,
        )

    def initialize_optimizer(self, registration):
        optimizer = torch.optim.Adam(
            [
                {"params": [registration.rotation], "lr": 7.5e-3},
                {"params": [registration.translation], "lr": 7.5e0},
            ],
            maximize=True,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=25,
            gamma=0.9,
        )
        return optimizer, scheduler

    def evaluate(self, registration):
        est_pose = registration.get_current_pose()
        rot = est_pose.get_rotation("euler_angles", "ZYX")
        xyz = est_pose.get_translation()
        alpha, beta, gamma = rot.squeeze().tolist()
        bx, by, bz = xyz.squeeze().tolist()
        param = [alpha, beta, gamma, bx, by, bz]
        geo = (
            torch.concat(
                [
                    *self.doublegeo(est_pose, self.pose),
                    self.geodesics(est_pose, self.pose),
                ]
            )
            .squeeze()
            .tolist()
        )
        tre = self.target_registration_error(est_pose.cpu()).item()
        return param, geo, tre

    def run(self, idx):
        img, pose = self.specimen[idx]
        img = self.transforms(img).to(self.device)
        self.pose = pose.to(self.device)

        registration = self.initialize_registration(img)
        optimizer, scheduler = self.initialize_optimizer(registration)
        self.target_registration_error = Evaluator(self.specimen, idx)

        # Initial loss
        param, geo, tre = self.evaluate(registration)
        params = [param]
        losses = []
        geodesic = [geo]
        fiducial = [tre]
        times = []

        itr = (
            tqdm(range(self.n_iters), ncols=75) if self.verbose else range(self.n_iters)
        )
        for _ in itr:
            t0 = time.perf_counter()
            optimizer.zero_grad()
            pred_img, mask = registration()
            loss = self.criterion(pred_img, img)
            loss.backward()
            optimizer.step()
            scheduler.step()
            t1 = time.perf_counter()

            param, geo, tre = self.evaluate(registration)
            params.append(param)
            losses.append(loss.item())
            geodesic.append(geo)
            fiducial.append(tre)
            times.append(t1 - t0)

        # Loss at final iteration
        pred_img, mask = registration()
        loss = self.criterion(pred_img, img)
        losses.append(loss.item())
        times.append(0)

        # Write results to dataframe
        df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
        df["ncc"] = losses
        df[["geo_r", "geo_t", "geo_d", "geo_se3"]] = geodesic
        df["fiducial"] = fiducial
        df["time"] = times
        df["idx"] = idx
        df["parameterization"] = self.parameterization
        return df


def main(id_number, parameterization):
    ckpt = torch.load(f"checkpoints/specimen_{id_number:02d}_best.ckpt")
    model = PoseRegressor(
        ckpt["model_name"],
        ckpt["parameterization"],
        ckpt["convention"],
        norm_layer=ckpt["norm_layer"],
    )
    model.load_state_dict(ckpt["model_state_dict"])

    specimen = DeepFluoroDataset(id_number)
    height = ckpt["height"]
    subsample = (1536 - 100) / height
    delx = 0.194 * subsample

    drr = DRR(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.focal_len / 2,
        height=height,
        delx=delx,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
        bone_attenuation_multiplier=2.5,
    )

    registration = Registration(
        drr,
        specimen,
        model,
        parameterization,
    )
    for idx in tqdm(range(len(specimen)), ncols=100):
        df = registration.run(idx)
        df.to_csv(
            f"runs/specimen{id_number:02d}_xray{idx:03d}_{parameterization}.csv",
            index=False,
        )


if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    id_numbers = [1, 2, 3, 4, 5, 6]
    parameterizations = [
        "se3_log_map",
        "so3_log_map",
        "axis_angle",
        "euler_angles",
        "quaternion",
        "rotation_6d",
        "rotation_10d",
        "quaternion_adjugate",
    ]
    id_numbers = [i for i, _ in product(id_numbers, parameterizations)]
    parameterizations = [p for _, p in product(id_numbers, parameterizations)]
    Path("runs").mkdir(exist_ok=True)

    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="registration",
        gpus_per_node=1,
        mem_gb=10.0,
        slurm_array_parallelism=12,
        slurm_partition="2080ti",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, id_numbers, parameterizations)
