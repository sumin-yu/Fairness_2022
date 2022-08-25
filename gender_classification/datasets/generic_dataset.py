from os.path import join
import numpy as np
import pandas as pd

import torch.utils.data as data
from os.path import expanduser


class GenericDataset(data.Dataset):
    def __init__(self, mri_root=None, split="train", transform=None, seed=0):
        self.mri_root_dir = mri_root
        self.num_classes = 2
        self.seed = seed
        self.split = split
        self.transform = transform

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def _split_mri_df(self, use_single_img=False, seed=0):
        label_map, target_labels = self._get_target_labels()
        df_mri = self._get_mri_df()

        df_mri.loc[:, "Gender"] = df_mri["PTGENDER"].map(lambda x: label_map[x])

        df_train, df_test = self._split_df_by_key(df_mri, key="SubjectID", ratio=0.8, seed=seed)
        df_train, df_val = self._split_df_by_key(df_train, key="SubjectID", ratio=0.8, seed=seed) # df_mri -> df_train 
        
        ids_to_be_removed = pd.Series(df_test["SubjectID"].unique(), name="SubjectID")
        ids_to_be_removed = np.concatenate([
            df_train["SubjectID"].unique(),
            df_val["SubjectID"].unique(),
            ids_to_be_removed.values
        ])

        total_ids = pd.Series(df_mri["SubjectID"].unique(), name="SubjectID")
        train_ids_series = total_ids[~total_ids.isin(ids_to_be_removed)]
        df_only_mri = df_mri.merge(train_ids_series, how="inner", on="SubjectID")
        df_only_mri_train, df_only_mri_val = self._split_df_by_key(df_only_mri, key="SubjectID",
                                                                    ratio=0.8, seed=seed)

        df_train = pd.concat([df_train, df_only_mri_train])
        df_val = pd.concat([df_val, df_only_mri_val])

        if use_single_img:
            df_train = df_train.groupby(["SubjectID"]).first().reset_index()
            df_val = df_val.groupby(["SubjectID"]).first().reset_index()
            df_test = df_test.groupby(["SubjectID"]).first().reset_index()
        
        return df_train, df_val, df_test

    @staticmethod
    def _get_mri_df():
        home = expanduser("~")
        df_mri = pd.read_csv(join(home, "data/fsdat_masterfile_20211027_new_only_QCed.csv"))
        return df_mri

    @staticmethod
    def _split_df_by_key(df, key="SubjectID", ratio=0.8, seed=1):
        """
        Split the DataFrame based on key
        """
        idx = df[key].unique()
        np.random.seed(seed)
        np.random.shuffle(idx)
        sub_idx_train = pd.Series(idx[: int(len(idx) * ratio)], name=key)
        sub_idx_test = pd.Series(idx[int(len(idx) * ratio):], name=key)
        df_train = df.merge(sub_idx_train, how="inner", on=key)
        df_test = df.merge(sub_idx_test, how="inner", on=key)
        return df_train, df_test

    @staticmethod
    def _get_target_labels():
        label_map = {"F": 1, "M": 0}
        target_labels = ["F", "M"]
        return label_map, target_labels
