import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import random as random
import os


class LibriMix(Dataset):
    """" Dataset class for Librimix (subset) source separation tasks.

    Args:
        root (str): The path to the dataset
            default: ``'../datasets/Avatar10Mix2'``
        task (str): One of ``'enh_single'``, ``'enh_both'``, ``'sep_clean'`` or
            ``'sep_noisy'``.

            * ``'enh_single'`` for single speaker speech enhancement.
            * ``'enh_both'`` for multi speaker speech enhancement.
            * ``'sep_clean'`` for two-speaker clean source separation.
            * ``'sep_noisy'`` for two-speaker noisy source separation.

        sample_rate (int) : The sample rate of the sources and mixtures
        n_src (int) : The number of sources in the mixture
        segment (int) : The desired sources and mixtures length in s

    References
        "LibriMix: An Open-Source Dataset for Generalizable Speech Separation",
        Cosentino et al. 2020.
    """

    dataset_name = "LibriMix"

    def __init__(self, root, task="sep_clean", sample_rate=16000,
                 n_src=2, segment=3, is_training=True):
        self.root = root
        csv_dir = os.path.join(root, "metadata")
        self.task = task
        self.sample_rate = sample_rate
        self.n_src = n_src
        self.segment = segment
        self.is_training = is_training

        # Get the csv corresponding to the task
        if task == "enh_single":
            flag = "%s_mix_single" % ("train" if is_training else "test")
        elif task == "sep_clean":
            flag = "%s_mix_clean" % ("train" if is_training else "test")
        elif task == "sep_noisy":
            flag = "%s_mix_both" % ("train" if is_training else "test")
        elif task == "enh_both":
            flag = "%s_mix_both" % ("train" if is_training else "test")
            clean_flag = "%s_mix_clean" % ("train" if is_training else "test")
            md_clean_file = [f for f in os.listdir(csv_dir) if clean_flag in f and f.startswith("mix")][0]
            self.df_clean = pd.read_csv(os.path.join(csv_dir, md_clean_file))
        else:
            raise ValueError("No such task!")

        md_file = [f for f in os.listdir(csv_dir) if flag in f and f.startswith("mix")][0]
        self.csv_path = os.path.join(csv_dir, md_file)

        # Data frame of mixtures
        self.df = pd.read_csv(self.csv_path)

        # Get rid of the utterances too short
        if self.segment is not None:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)
            # Ignore the file shorter than the desired_length
            self.df = self.df[self.df["length"] >= self.seg_len]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.seg_len = None

        # Set of speakers
        spk_names = np.unique([i.split('-')[0] for i in self.df['mixture_ID']])
        self.spk_dict = {n:i for i, n in enumerate(spk_names)}

    def subset_indices(self, n_shot):
        """ Retrieve n shots for each speaker
        """
        indices = []
        for name in self.spk_dict.keys():
            cnt = 0
            for index, row in self.df.iterrows():
                if cnt >= n_shot:
                    break
                if row['mixture_ID'].split('-')[0] == name:
                    cnt += 1
                    indices.append(index)
        return indices

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row in dataframe
        row = self.df.iloc[idx]

        # Get speaker ids
        mixture_id = row['mixture_ID']
        spk_ids = mixture_id.split('_')
        spk_ids = [int(self.spk_dict[ii.split('-')[0]]) for ii in spk_ids]

        # If there is a seg start point o.w. set randomly
        if self.seg_len is not None:
            if self.is_training:
                start = random.randint(0, row["length"] - self.seg_len)
            else:
                start = 0
            stop = start + self.seg_len
        else:
            start = 0
            stop = None

        # Read sources
        sources_list = []
        if "enh_both" in self.task:
            mix_clean_path = self.df_clean.iloc[idx]["mixture_path"]
            mix_clean_path = os.path.join(self.root, mix_clean_path)
            s, _ = sf.read(mix_clean_path, dtype="float32", start=start, stop=stop)
            # If task is enh_both then the source is the clean mixture
            sources_list.append(s)
        elif self.task == 'enh_single':
            source_path = row[f"source_1_path"]
            source_path = os.path.join(self.root, source_path)
            s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
            sources_list.append(s)
        else:
            for i in range(self.n_src):
                source_path = row[f"source_{i + 1}_path"]
                source_path = os.path.join(self.root, source_path)
                s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
                sources_list.append(s)

        # Read the mixture
        mixture_path = os.path.join(self.root, row["mixture_path"])
        mixture, _ = sf.read(mixture_path, dtype="float32", start=start, stop=stop)
        # Convert to torch tensor
        mixture = torch.from_numpy(mixture)
        # Stack sources
        sources = np.vstack(sources_list) # n_src x seg_len

        if self.task == 'enh_single':
            sources = sources[0]
            spk_ids = spk_ids[0]

        # Convert sources to tensor
        sources = torch.from_numpy(sources)

        return (mixture, spk_ids), sources

    def _dataset_name(self):
        """ Differentiate between 2 and 3 sources."""
        return f"Avatar{len(self.spk_dict)}Mix{self.n_src}"


if __name__ == '__main__':
    root = '../datasets/Avatar10Mix2'
    dataset = LibriMix(root, task="enh_single", sample_rate=8000,
                       segment=None, is_training=False)

    (mixture, spk_ids), sources = dataset[1]
    print(mixture.shape, sources.shape, spk_ids)
