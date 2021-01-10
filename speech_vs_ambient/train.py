import os
import argparse
import json
import sys

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.schedulers import DPTNetScheduler
from asteroid.engine.system import System
from asteroid.losses import SingleSrcNegSDR

sys.path.append("../")

from src.librimix_dataset import LibriMix
from src.model import AvaTr


# Additional arguments
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")


def main(conf):
    if conf["data"]["task"] == 'enh_single':
        conf["separator"]["n_src"] = 1
    else:
        conf["separator"]["n_src"] = conf["data"]["n_src"]
    conf["filterbank"]["sample_rate"] = conf["data"]["sample_rate"] 

    # Save args
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Data loaders
    train_set = LibriMix(
        conf["data"]["root"],
        conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
        is_training=True
    )
    val_set = LibriMix(
        conf["data"]["root"],
        conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
        is_training=False
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )

    # Model
    model = AvaTr(**conf["avatar"], **conf["separator"], **conf["filterbank"])

    # Loss function
    loss_func = SingleSrcNegSDR("sisdr", reduction='mean')

    # Optimizer
    optimizer = make_optimizer(model.parameters(), **conf["optim"])

    # lr scheduler
    if conf["training"]["lr_scheduler"] == "plateau":
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=30, verbose=True)
    elif conf["training"]["lr_scheduler"] == "dpt":
        scheduler = {
            "scheduler": DPTNetScheduler(
                optimizer, len(train_loader) // conf["training"]["batch_size"], 64
            ),
            "interval": "step",
        }
    else:
        scheduler = None

    # Callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

    # Trainer
    system = System(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=conf,
    )
    gpus = conf["training"]["gpus"] if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None
    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        gpus=gpus,
        distributed_backend=distributed_backend,
        gradient_clip_val=conf["training"]["gradient_clipping"],
    )

    #(mix, sid), src = next(iter(train_loader))
    #mix, sid = mix.to('cuda:0'), sid.to('cuda:0')
    #model = model.to('cuda:0')
    #est = model.forward((mix, sid))

    # Training
    trainer.fit(system)

    # Save
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # Load configs
    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)

    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic = parse_args_as_dict(parser, return_plain_args=False)
    pprint(arg_dic)
    main(arg_dic)
