from torchvision import transforms as transform_lib
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
import torch
from custom_data import CelebADataset
from typing import Any, Callable, Optional
import os

class CelebADataModule(LightningDataModule):  # pragma: no cover
    """
    .. 
        :width: 400
        # :alt: STL-10
    Specs:
        - 10 classes (1 per type)
        # - Each image is (3 x 96 x 96)
    Standard STL-10, train, val, test splits and transforms.
    STL-10 has support for doing validation splits on the labeled or unlabeled splits
    Transforms::
        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            transforms.Normalize(
                mean=(0.43, 0.42, 0.39),
                std=(0.27, 0.26, 0.27)
            )
        ])
    Example::
        from pl_bolts.datamodules import STL10DataModule
        dm = STL10DataModule(PATH)
        model = LitModel()
        Trainer().fit(model, datamodule=dm)
    """

    name = "CelebA"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_size: float = 0.2,
        num_workers: int = 0,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: where to save/load the data
            num_workers: how many workers to use for loading data
            batch_size: the batch size
            seed: random seed to be used for train/val/test splits
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(*args, **kwargs)

        self.dims = (3, 224, 224)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.val_size = val_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.num_samples = int(len(os.listdir())*(1-val_size))


    # def prepare_data(self) -> None:
    #     """Downloads the unlabeled, train and test split."""
        # STL10(self.data_dir, split="unlabeled", download=True, transform=transform_lib.ToTensor())
        # STL10(self.data_dir, split="train", download=True, transform=transform_lib.ToTensor())
        # STL10(self.data_dir, split="test", download=True, transform=transform_lib.ToTensor())

    def train_dataloader(self) -> DataLoader:
        """Loads the 'unlabeled' split minus a portion set aside for validation via `unlabeled_val_split`."""
        transforms = self._default_transforms() if self.train_transforms is None else self.train_transforms

        dataset = CelebADataset(self.data_dir, transform=transforms)
        val_length = int(len(dataset)*self.val_size)
        train_length = len(dataset) - val_length
        dataset_train, _ = random_split(
            dataset,
            [train_length, val_length],
            generator=torch.Generator().manual_seed(self.seed),
        )
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        """Loads a portion of the 'unlabeled' training data set aside for validation.
        The val dataset = (unlabeled - train_val_split)
        Args:
            batch_size: the batch size
            transforms: a sequence of transforms
        """
        transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms
        dataset = CelebADataset(self.data_dir, transform=transforms)
        val_length = int(len(dataset)*self.val_size)
        train_length = len(dataset) - val_length
        _, dataset_val = random_split(
            dataset,
            [train_length, val_length],
            generator=torch.Generator().manual_seed(self.seed),
        )

        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def _default_transforms(self) -> Callable:
        data_transforms = transform_lib.Compose([transform_lib.Resize((224,224)), transform_lib.ToTensor(), transform_lib.Normalize(0.5, 0.5)])
        return data_transforms


if __name__ == "__main__":
    
    dm = CelebADataModule("data/CelebA/img_align_celeba")
    x = next(iter(dm.train_dataloader()))

    # print(x.max())
    print(x.shape)