import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def fetch_dataloader(types, data_dir, download, params):
    """
    Fetches DataLoader for the MNIST dataset for type in types.

    Parameters
    ----------
    types: list
        the type of DataLoaders to fetch.

    data_dir: str
        path to the root of the dataset

    download: bool
        True if the MNIST dataset is not downloaded

    params: Params
        params for the DataLoader

    Returns
    -------
    a dict containing the DataLoaders in types

    """

    dataloaders = {}

    for split in ['train', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_signs".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':

                dl = DataLoader(dataset=datasets.MNIST(root=data_dir,
                                                       download=download,
                                                       train=True,
                                                       transform=transforms.Compose([
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.1307,),
                                                                                (0.3081,)),
                                                       ])),
                                batch_size=params.batch_size,
                                shuffle=True,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)
            else:
                dl = DataLoader(dataset=datasets.MNIST(root=data_dir,
                                                       download=download,
                                                       train=False,
                                                       transform=transforms.Compose([
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.1307,),
                                                                                (0.3081,)),
                                                       ])),
                                batch_size=params.batch_size,
                                shuffle=True,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
