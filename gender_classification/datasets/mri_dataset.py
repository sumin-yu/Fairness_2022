from os.path import join
import nibabel as nib

from monai.transforms import AddChannel, Compose, ScaleIntensity, EnsureType, Pad
from datasets.generic_dataset import GenericDataset


class MRIDataset(GenericDataset):
    def __init__(
        self,
        root,
        split="train",
        seed=0,
        use_single_img=False
    ):
        transform = Compose([Pad([(3, 2), (0, 0), (3, 2)]), ScaleIntensity(), AddChannel(), EnsureType()])
        super(MRIDataset, self).__init__(mri_root=root, split=split, transform=transform, seed=seed)

        df_train, df_val, df_test = self._split_mri_df(
            use_single_img=use_single_img, seed=seed
        )

        if self.split == "train":
            self.df = df_train
        elif self.split == "val":
            self.df = df_val
        else:
            self.df = df_test

        self.df.reset_index(inplace=True)
        self.use_single_img = use_single_img
        self.labels = df_train["Gender"]

    def __getitem__(self, idx):
        weight = 0
        img_names = self.df.loc[idx, "File_name"]
        label = self.df.loc[idx, "Gender"]
        sub_id = self.df.loc[idx, "SubjectID"]

        img_path = join(self.mri_root_dir, img_names, "brain_to_MNI_nonlin.nii.gz")
        img = nib.load(img_path).get_fdata()

        if self.transform:
            img = self.transform(img)
            img = img[:, :, 6:102, :]

        return img, label, sub_id, idx

    def __len__(self):
        return len(self.df)
