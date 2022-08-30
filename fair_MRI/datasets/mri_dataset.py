from os.path import join
import nibabel as nib

from monai.transforms import Pad, AddChannel, Compose, ScaleIntensity, EnsureType, RandAdjustContrast, RandGaussianSmooth
# from torchio.transforms import Compose, OneOf, Pad, Resample, RescaleIntensity, RandomBiasField, RandomSpike, RandomMotion, RandomGhosting
from datasets.generic_dataset import GenericDataset


class MRIDataset(GenericDataset):
    def __init__(
        self,
        root,
        split="train",
        seed=0,
        with_mci=False,
        use_single_img=False,
        method = 'None',
        aug=False
    ):
        if aug:
            train_transform = Compose([Pad([(3, 2), (0, 0), (3, 2)]), ScaleIntensity(), RandAdjustContrast(prob=0.4), RandGaussianSmooth(prob=0.3), AddChannel(), EnsureType()])
        else:
            train_transform = Compose([Pad([(3, 2), (0, 0), (3, 2)]), ScaleIntensity(), AddChannel(), EnsureType()])
        test_transform = Compose([Pad([(3, 2), (0, 0), (3, 2)]), ScaleIntensity(), AddChannel(), EnsureType()])

        # pad = Pad(3,2,0,0,3,2)
        # resample = Resample()
        # scaleintensity = RescaleIntensity()
        # biasfield = RandomBiasField()
        # spike = RandomSpike()
        # motion = RandomMotion()
        # ghosting = RandomGhosting()
        # transform = OneOf({biasfield:0.25, spike:0.25, motion:0.25, ghosting:0.25})
        if split=="train":
            super(MRIDataset, self).__init__(mri_root=root, split=split, transform=train_transform, seed=seed, with_mci=with_mci, method=method)
        else :
            super(MRIDataset, self).__init__(mri_root=root, split=split, transform=test_transform, seed=seed, with_mci=with_mci, method=method)


        df_train, df_val, df_test = self._split_mri_df(
            with_mci=with_mci, use_single_img=use_single_img, seed=seed
        )

        if self.split == "train":
            self.df = df_train
        elif self.split == "val":
            self.df = df_val
        else:
            self.df = df_test

        self.df.reset_index(inplace=True)
        self.use_single_img = use_single_img
        self.method = method
        self.labels = df_train["Dx_num"]
        self.groups = df_train["Gender_num"] 

    def make_weights(self, method='None'):
        if method == 'oversampling':
            group_weights = len(self) / self.num_data
            weights = [group_weights[int(self.groups[i]), int(y)] for i, y in enumerate(self.labels)]
        return weights

    def __getitem__(self, idx):
        weight = 0
        img_names = self.df.loc[idx, "File_name"]
        label = self.df.loc[idx, "Dx_num"]
        sub_id = self.df.loc[idx, "SubjectID"]
        #########
        gender = self.df.loc[idx, "Gender_num"]

        img_path = join(self.mri_root_dir, img_names, "brain_to_MNI_nonlin.nii.gz")
        img = nib.load(img_path).get_fdata()
        # print(img.shape)
        if self.transform:
            img = self.transform(img)
            img = img[:, :, 6:102, :]
        # print(img.shape)
        if gender ==0 and label ==1:
            weight = self.weights_m[0]
        elif gender ==0 and label ==0:
            weight = self.weights_m[1]
        elif gender ==1 and label == 1:
            weight = self.weights_m[2]
        else : weight = self.weights_m[3]
        # print(self.weights_m)
        return img, label, sub_id, idx, gender, weight

    def __len__(self):
        return len(self.df)
