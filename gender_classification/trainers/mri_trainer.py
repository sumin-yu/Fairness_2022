from __future__ import print_function
from datasets.mri_dataset import MRIDataset
from networks.mobilenetv2 import MobileNetV2
from networks.shufflenetv2 import ShuffleNetV2
from networks.resnet import resnet18, resnet10

from trainers.generic_trainer import GenericTrainer
from os.path import join, expanduser


class Trainer(GenericTrainer):
    def __init__(self, args, log_dir=None, log_name=None, save_dir=None, **kwargs):
        super().__init__(args, log_dir, log_name, save_dir, **kwargs)

    def _get_dataset(self, args):
        mri_image_root = join(expanduser("~"), "./data/adni_registration_all/")
        train_dataset = MRIDataset(mri_image_root, split="train", seed=args.subject_seed,
                                   use_single_img=args.use_single_img
                                   )
        val_dataset = MRIDataset(mri_image_root, split="val", seed=args.subject_seed, 
                                 use_single_img=args.use_single_img
                                 )
        test_dataset = MRIDataset(mri_image_root, split="test", seed=args.subject_seed,
                                  use_single_img=args.use_single_img
                                  )
        return train_dataset, val_dataset, test_dataset

    def _init_model(self, args):
        num_outputs = 1
        if args.network == 'mobilenetv2':
            self.model = MobileNetV2(num_classes=num_outputs, sample_size=96, in_channel=1)
        elif args.network == 'shufflenetv2' :
            self.model = ShuffleNetV2(num_classes=num_outputs, sample_size=96, in_channel=1)
            # self.model = MobileNetV2(num_classes=num_outputs, sample_size=96, in_channel=1)
            # self.model = resnet10(pretrained=args.pretrained, num_classes=num_outputs, n_input_channels=1)

        if args.cuda:
            self.model = self.model.cuda()
