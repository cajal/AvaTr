import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint

from asteroid.metrics import get_metrics

import sys
sys.path.append("../")

from src.librimix_dataset import LibriMix
from src.model import AvaTr
from src.vis import to_html_single
from src.utils import tensors_to_device

parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_dir", type=str,
    default="exp/tmp",
    help="Experiment root")
parser.add_argument(
    "--out_dir", type=str,
    default='vis',
    help="Directory in exp_dir where the eval results" " will be stored",
)
parser.add_argument(
    "--use_gpu", type=int, default=1, help="Whether to use the GPU for model execution"
)

compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]


def main(args, conf):
    model_path = os.path.join(args["exp_dir"], "best_model.pth")
    model = AvaTr(**conf["avatar"],
                  **conf["separator"],
                  **conf["filterbank"])
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])

    # Handle device placement
    if args["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device

    test_set = LibriMix(
        conf["data"]["root"],
        conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=None, # Uses all segment length
        is_training=False
    )

    # Randomly choose the indexes of sentences to save.
    eval_save_dir = os.path.join(args["exp_dir"], args["out_dir"])
    ex_save_dir = os.path.join(eval_save_dir, "examples/")

    series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        mixture_id = test_set.df.iloc[idx]['mixture_ID']

        # Forward the network on the mixture.
        (mix, ids), sources = tensors_to_device(test_set[idx], device=model_device)
        ids = torch.LongTensor([ids]).to(model_device)
        est_sources = model((mix.unsqueeze(0), ids.unsqueeze(0)))

        mix_np = mix.cpu().data.numpy()
        sources_np = sources.cpu().data.numpy()
        est_sources_np = est_sources.squeeze(0).cpu().data.numpy()

        # Eval
        utt_metrics = get_metrics(
            mix_np,
            sources_np,
            est_sources_np,
            sample_rate=conf["data"]["sample_rate"],
            metrics_list=compute_metrics,
        )
        utt_metrics["mix_id"] = mixture_id
        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        int_sisdr = int(utt_metrics['si_sdr'])
        save_flag = False
        if int_sisdr >= 14:
            save_flag = True

        if save_flag:
            local_save_dir = os.path.join(ex_save_dir, "ex_{}_{:.2f}/".format(mixture_id, utt_metrics['si_sdr']))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "mixture.wav", mix_np, conf["data"]["sample_rate"])

            # Loop over the sources and estimates
            src = sources_np
            sf.write(local_save_dir + "s{}.wav".format(1), src, conf["data"]["sample_rate"])

            est_src = est_sources_np
            est_src *= np.max(np.abs(mix_np)) / np.max(np.abs(est_src))
            sf.write(
                local_save_dir + "s{}_estimate.wav".format(1),
                est_src,
                conf["data"]["sample_rate"],
            )
            # Write local metrics to the example folder.
            with open(local_save_dir + "metrics.json", "w") as f:
                json.dump(utt_metrics, f, indent=0)

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()
    print("Overall metrics :")
    pprint(final_results)
    with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)

    # HTML
    to_html_single(ex_save_dir)


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)

    main(arg_dic, train_conf)
