from os.path import join
import numpy as np
import pandas as pd

import torch.utils.data as data
from os.path import expanduser


class GenericDataset(data.Dataset):
    def __init__(self, mri_root=None, split="train", transform=None, seed=0, with_mci=False, method='None'):
        self.mri_root_dir = mri_root
        self.num_classes = 3 if with_mci else 2
        self.seed = seed
        self.with_mci = with_mci
        self.method = method
        self.split = split
        self.transform = transform
        self.weights_m = None
        self.num_data = np.zeros((2,self.num_classes))

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def _split_mri_df(self, with_mci=False, use_single_img=False, seed=0):
        label_map, target_labels = self._get_target_labels(with_mci=with_mci)
        df_mri = self._get_mri_df()

        df_mri = df_mri.query("Dx_new == @target_labels")

        df_mri.loc[:, "Dx_num"] = df_mri["Dx_new"].map(lambda x: label_map[x])
        df_mri["Gender_num"] = [int(0) if s == 'M' else int(1) for s in df_mri['PTGENDER']]

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
        
        m = len(df_train.loc[(df_train["PTGENDER"]=='M')])
        f = len(df_train.loc[(df_train["PTGENDER"]=='M')])
        ad = len(df_train["Dx_new"]=='AD')
        cn = len(df_train["Dx_new"]=='CN')
        ad_m = len(df_train.loc[(df_train["PTGENDER"]=='M') & (df_train["Dx_new"]=='AD')])
        cn_m = len(df_train.loc[(df_train["PTGENDER"]=='M') & (df_train["Dx_new"]=='CN')])
        ad_f = len(df_train.loc[(df_train["PTGENDER"]=='F') & (df_train["Dx_new"]=='AD')])
        cn_f = len(df_train.loc[(df_train["PTGENDER"]=='F') & (df_train["Dx_new"]=='CN')])
        if self.method == "WCE":
            tot = 1/ad_m + 1/cn_m + 1/ad_f + 1/cn_f
            self.weights_m = np.array([1/(tot*ad_m),1/(tot*cn_m),1/(tot*ad_f),1/(tot*cn_f)])
        else: # method == RW
            tot = ad*m/ad_m + cn*m/cn_m + ad*f/ad_f + cn*f/cn_f
            self.weights_m = np.array([ad*m/(tot*ad_m),cn*m/(tot*cn_m),ad*f/(tot*ad_f),cn*f/(tot*cn_f)])
        self.num_data[0,0]=cn_m
        self.num_data[0,1]=ad_m
        self.num_data[1,0]=cn_f
        self.num_data[1,1]=ad_f

        ########## data analysis ############
        # single = 'no-single'
        # if use_single_img:
        #     single = 'use-single'
        # writer = pd.ExcelWriter('./data_anal/subject-seed={}&{}.xlsx'.format(seed,single), engine='openpyxl')
        # df_train[['SubjectID','Dx_new','PTGENDER']].to_excel(writer, sheet_name='train')
        # df_val[['SubjectID','Dx_new','PTGENDER']].to_excel(writer, sheet_name='val')
        # df_test[['SubjectID','Dx_new','PTGENDER']].to_excel(writer, sheet_name='test')
        # writer.save()

        return df_train, df_val, df_test

    @staticmethod
    def _get_mri_df():
        home = expanduser("~")
        df_mri = pd.read_csv(join(home, "data/fsdat_masterfile_20211027_new_only_QCed.csv"))
        df_mri = df_mri.rename({"Dx.new": "Dx_new"}, axis="columns")
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
    def _get_target_labels(with_mci=False):
        if with_mci:
            label_map = {"AD": 2, "MCI": 1, "CN": 0}
            target_labels = ["AD", "MCI", "CN"]

        else:
            label_map = {"AD": 1, "CN": 0}
            target_labels = ["AD", "CN"]
        return label_map, target_labels
