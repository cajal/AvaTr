import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import random as random
import os


class LibriMix(Dataset):
    """" Dataset class for Librimix source separation tasks.

    Args:
        csv_dir (str): The path to the metatdata file
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

    def __init__(self, csv_dir, task="sep_clean", sample_rate=16000,
                 n_src=2, segment=3, is_training=True):
        self.csv_dir = csv_dir
        self.task = task
        self.sample_rate = sample_rate
        self.n_src = n_src
        self.segment = segment
        self.is_training = is_training

        # Get the csv corresponding to the task
        if task == "enh_single":
            md_file = [f for f in os.listdir(csv_dir) if "single" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        elif task == "enh_both":
            md_file = [f for f in os.listdir(csv_dir) if "both" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
            md_clean_file = [f for f in os.listdir(csv_dir) if "clean" in f][0]
            self.df_clean = pd.read_csv(os.path.join(csv_dir, md_clean_file))
        elif task == "sep_clean":
            md_file = [f for f in os.listdir(csv_dir) if "clean" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        elif task == "sep_noisy":
            md_file = [f for f in os.listdir(csv_dir) if "both" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)

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

        # Get mixture path
        self.mixture_path = row["mixture_path"]

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
            s, _ = sf.read(mix_clean_path, dtype="float32", start=start, stop=stop)
            # If task is enh_both then the source is the clean mixture
            sources_list.append(s)
        elif self.task == 'enh_single':
            source_path = row[f"source_1_path"]
            s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
            sources_list.append(s)
        else:
            for i in range(self.n_src):
                source_path = row[f"source_{i + 1}_path"]
                s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
                sources_list.append(s)

        # Read the mixture
        mixture, _ = sf.read(self.mixture_path, dtype="float32", start=start, stop=stop)
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
    csv_dir = 'data/wav8k/min/train-100-rand10-train'
    dataset = LibriMix(csv_dir, task="enh_single", sample_rate=8000)

    (mixture, spk_ids), sources = dataset[1]
    print(mixture.shape, sources.shape, spk_ids)
