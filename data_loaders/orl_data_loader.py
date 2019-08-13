import os
from PIL import Image
import pickle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


transformer = transforms.Compose([transforms.ToTensor()])


class ORLDataset(Dataset):

    def __init__(self, pkl_file, transform):
        self.filenames = self.fetch_filenames(pkl_file)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        face = Image.open(self.filenames[index])
        face = self.transform(face)
        return face

    def fetch_filenames(self, split):
        with open(split, 'rb') as f:
            filenames = pickle.load(f)
        return filenames


def fetch_dataloader(types, data_dir, params):
    dataloaders = {}

    for split in ['train', 'test']:
        if split in types:
            pkl_file = os.path.join(data_dir, split)
            dl = DataLoader(ORLDataset(pkl_file, transformer),
                            batch_size=params.batch_size,
                            shuffle=True,
                            num_workers=params.num_workers,
                            pin_memory=params.cuda,
                            drop_last=True)

            dataloaders[split] = dl

    return dataloaders

