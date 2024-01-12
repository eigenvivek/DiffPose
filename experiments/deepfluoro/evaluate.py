from pathlib import Path

import pandas as pd
import submitit
import torch
from tqdm import tqdm

from diffpose.deepfluoro import DeepFluoroDataset, Evaluator, Transforms
from diffpose.registration import PoseRegressor


def load_specimen(id_number, device):
    specimen = DeepFluoroDataset(id_number)
    isocenter_pose = specimen.isocenter_pose.to(device)
    return specimen, isocenter_pose


def load_model(model_name, device):
    ckpt = torch.load(model_name)
    model = PoseRegressor(
        ckpt["model_name"],
        ckpt["parameterization"],
        ckpt["convention"],
        norm_layer=ckpt["norm_layer"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    transforms = Transforms(ckpt["height"])
    return model, transforms


def evaluate(specimen, isocenter_pose, model, transforms, device):
    error = []
    model.eval()
    for idx in tqdm(range(len(specimen)), ncols=100):
        target_registration_error = Evaluator(specimen, idx)
        img, _ = specimen[idx]
        img = img.to(device)
        img = transforms(img)
        with torch.no_grad():
            offset = model(img)
        pred_pose = isocenter_pose.compose(offset)
        mtre = target_registration_error(pred_pose.cpu()).item()
        error.append(mtre)
    return error


def main(id_number):
    device = torch.device("cuda")
    specimen, isocenter_pose = load_specimen(id_number, device)
    models = sorted(Path("checkpoints/").glob(f"specimen_{id_number:02d}_epoch*.ckpt"))

    errors = []
    for model_name in models:
        model, transforms = load_model(model_name, device)
        error = evaluate(specimen, isocenter_pose, model, transforms, device)
        errors.append([model_name.stem] + error)

    df = pd.DataFrame(errors)
    df.to_csv(f"evaluations/subject{id_number}.csv", index=False)


if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    Path("evaluations").mkdir(exist_ok=True)
    id_numbers = [1, 2, 3, 4, 5, 6]

    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="eval",
        gpus_per_node=1,
        mem_gb=10.0,
        slurm_array_parallelism=len(id_numbers),
        slurm_exclude="curcum",
        slurm_partition="2080ti",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, id_numbers)
