import argparse


def train_argument_parser():
    """
    A method for parsing arguments passed via the command line for training a Variational Autoencoder. The default
    arguments will load a fully connected Variational Autoencoder that will be trained on the MNIST dataset.

    Returns
    -------
    Returns a argparse object.

    """
    parser = argparse.ArgumentParser(description='Train Variational Autoencoder')

    parser.add_argument('--data_dir',
                        nargs="?",
                        default='data/mnist',
                        help="Directory containing the dataset")

    parser.add_argument('--model_dir',
                        nargs="?",
                        default='experiments/info_vae',
                        help="Directory containing params.json")

    parser.add_argument('--restore_file',
                        default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'train'

    parser.add_argument('--dataloader',
                        default='mnist',
                        help='Specify the dataloader to use')

    parser.add_argument('--loss',
                        default='bce',
                        help="Specifiy the reconstruction loss to use, i.e. bce, mse, mmd (InfoVAE)")

    parser.add_argument('--swa',
                        action='store_true',
                        help='Enables training with Stochastic Weight Averaging')

    return parser.parse_args()

def generate_argument_parser():
    """

    Returns
    -------

    """

    parser = argparse.ArgumentParser(description='Generate images with a Variational Autoencoder')

    parser.add_argument('--data_dir',
                        nargs="?",
                        default='data/mnist',
                        help="Directory containing the dataset")

    parser.add_argument('--model_dir',
                        nargs="?",
                        default='experiments/vae',
                        help="Directory containing params.json")

    parser.add_argument('--restore_file',
                        default='last',
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'train'

    parser.add_argument('--dataloader',
                        default='mnist',
                        help='Specify the dataloader to use')

    parser.add_argument('--run',
                        nargs='?',
                        default=None,
                        help='Specify the datetime of the run')

    parser.add_argument('--generate',
                        default='gif',
                        help='Specify what type of images to generatemodel')

    return parser.parse_args()


