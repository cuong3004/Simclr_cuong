from torchvision import transforms
import math
import numpy as np
import cv2
import torchvision
import matplotlib.pyplot as plt

def test_transform(dataloader):
    plt.figure(figsize=(10,10))
    x1, _, _ = next(iter(dataloader))
    x1 = x1[:5*5]
    grid_img = torchvision.utils.make_grid(x1, nrow=5)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig("test_image.png")


# warmup + decay as a function
def linear_warmup_decay(warmup_steps, total_steps, cosine=True, linear=False):
    """Linear warmup for warmup_steps, optionally with cosine annealing or linear decay to 0 at total_steps."""
    assert not (linear and cosine)

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        if not (cosine or linear):
            # no decay
            return 1.0

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        if cosine:
            # cosine decay
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # linear decay
        return 1.0 - progress

    return fn




class SimCLRTrainDataTransform:
 

    def __init__(
        self,  train_transform
    ) -> None:
        self.train_transform = train_transform



    def __call__(self, sample):
        transform = self.train_transform

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj


class SimCLREvalDataTransform(SimCLRTrainDataTransform):
    """Transforms for SimCLR.
    Transform::
        Resize(input_height + 10, interpolation=3)
        transforms.CenterCrop(input_height),
        transforms.ToTensor()
    Example::
        from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform
        transform = SimCLREvalDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
        self, input_height: int = 224, gaussian_blur: bool = True, jitter_strength: float = 1.0, normalize=None
    ):
        super().__init__(
            normalize=normalize, input_height=input_height, gaussian_blur=gaussian_blur, jitter_strength=jitter_strength
        )

        # replace online transform with eval time transform
        self.online_transform = transforms.Compose(
            [
                transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
                transforms.CenterCrop(self.input_height),
                self.final_transform,
            ]
        )


