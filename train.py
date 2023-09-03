import argparse
import os
import warnings

import torch

from settings.config import Config
from src.dataset import FaceRecognitionDataset
from src.model import SNN
from src.utils import Trainer, get_transforms, set_seed

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="model")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    set_seed(Config.seed)

    backbone_mobile = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True)
    snn_mobile = SNN(backbone_mobile)

    dataset = FaceRecognitionDataset(Config.data_path, get_transforms())
    train_ds, val_ds = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(snn_mobile.parameters(), lr=Config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6, verbose=True)
    trainer = Trainer(
        snn_mobile,
        criterion,
        optimizer,
        train_dataloader,
        val_dataloader,
        Config.epochs,
        Config.device,
        lr_scheduler=lr_scheduler,
        model_name=args.model_name,
    )
    trainer.fit()

    best_model_path = [model for model in os.listdir(Config.model_path) if args.model_name in model and model.endswith(".pth")][-1]
    snn_mobile.load_state_dict(torch.load(os.path.join(Config.model_path, best_model_path)))
    snn_mobile.eval()

    img1 = torch.randn(1, 3, Config.resize_to[0], Config.resize_to[1], requires_grad=True)
    img2 = torch.randn(1, 3, Config.resize_to[0], Config.resize_to[1], requires_grad=True)
    torch_out = snn_mobile(img1, img2)

    torch.onnx.export(
        snn_mobile,
        (img1, img2),
        os.path.join(Config.model_path, f"{args.model_name}.onnx"),
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input1, input2"],
        output_names=["output"],
        dynamic_axes={"input1": {0: "batch_size"}, "input2": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
