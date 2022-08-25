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
        with_mci=False,
        use_single_img=False,
        method = 'None'
    ):
        transform = Compose([Pad([(3, 2), (0, 0), (3, 2)]), ScaleIntensity(), AddChannel(), EnsureType()])
        super(MRIDataset, self).__init__(mri_root=root, split=split, transform=transform, seed=seed, with_mci=with_mci)

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
        gender = self.df.loc[idx, "PTGENDER"]

        img_path = join(self.mri_root_dir, img_names, "brain_to_MNI_nonlin.nii.gz")
        img = nib.load(img_path).get_fdata()

        if self.transform:
            img = self.transform(img)
            img = img[:, :, 6:102, :]
        
        if gender =='M' and label ==1:
            weight = self.weights_m[0]
        elif gender =='M' and label ==0:
            weight = self.weights_m[1]
        elif gender =='F' and label == 1:
            weight = self.weights_m[2]
        else : weight = self.weights_m[3]
        # print(self.weights_m)
        return img, label, sub_id, idx, gender, weight

    def __len__(self):
        return len(self.df)
