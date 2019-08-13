"""Generate images with the Variational Autoencoder"""
import os
import sys
import glob
import imageio
import logging
import torch
import torchvision
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from datetime import datetime
from tqdm import trange

sys.path.append("..")
from utils import misc
from utils.parsers import generate_argument_parser


def show_images(images):
    """Transforms multiple images into a grid.

    Code referenced from http://hameddaily.blogspot.com/2018/12/yet-another-tutorial-on-variational.html

    Parameters
    ----------
    images

    Returns
    -------

    """
    images = torchvision.utils.make_grid(images)
    show_image(images[0])


def show_image(img):
    """Presents the generated images in a window.

    Code referenced from http://hameddaily.blogspot.com/2018/12/yet-another-tutorial-on-variational.html

    Parameters
    ----------
    img

    Returns
    -------

    """
    plt.imshow(img, cmap='gray')
    plt.show()


def generate_images(model, model_dir, dataloader, params):
    log_dir = misc.create_log_dir(model_dir, generated_img=True)
    with torch.no_grad() and SummaryWriter(log_dir) as writer:
        for i, (test_batch, _) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                test_batch = test_batch.cuda()
            # compute model output and loss
            X_reconstructed, mu, logvar, z = model(test_batch)

            if i % 1 == 0:
                X_reconstructed = X_reconstructed.view(params.batch_size, 1, 28, 28).cpu()
                images = torchvision.utils.make_grid(X_reconstructed)
                writer.add_image("generated_images_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), images, i)


def reconstruct_images_form_test_set(model, dataloader, dl_type, params, log_dir):
    # set model to evaluation mode
    model.eval()

    test_dir = os.path.join(log_dir, 'generated_images', 'test')
    misc.create_dir(test_dir)

    tf_logger = glob.glob(os.path.join(log_dir, '*.kompjuter'))

    ground_truth_dir = os.path.join(test_dir, 'ground_truth')
    misc.create_dir(ground_truth_dir)
    reconstructed_dir = os.path.join(test_dir, 'reconstructed')
    misc.create_dir(reconstructed_dir)

    with torch.no_grad() and SummaryWriter(tf_logger) as writer:
        training_progressor = trange(len(dataloader), desc="Reconstructing Test Set")
        for i in training_progressor:

            if dl_type == 'orl_face':
                test_batch = next(iter(dataloader))
            else:
                test_batch, _ = next(iter(dataloader))

            # move to GPU if available
            if params.cuda:
                test_batch = test_batch.cuda()

            X_reconstructed, z = model(test_batch)
            X_reconstructed = X_reconstructed.cpu()

            if i % 1 == 0:
                ground_truth_images = torchvision.utils.make_grid(test_batch.cpu())
                ground_truth_filename = os.path.join(ground_truth_dir, 'image_at_iter_{:06d}.png'.format(i))
                torchvision.utils.save_image(ground_truth_images, ground_truth_filename)

                reconstructed_images = torchvision.utils.make_grid(X_reconstructed)
                reconstructed_filename = os.path.join(reconstructed_dir, 'image_at_iter_{:06d}.png'.format(i))
                torchvision.utils.save_image(reconstructed_images, reconstructed_filename)

        logging.info("\nGenerating gif of ground truth.")
        generate_gif(ground_truth_dir, tbx_writer=writer)
        logging.info("Generating gif of reconstructed images.\n")
        generate_gif(reconstructed_dir, tbx_writer=writer)


def generate_and_save_images(model, epoch, test_input, img_dir):
    # reconstruct image batch from a latent vector called test_input
    X_reconstructed = model.sample(test_input)
    X_reconstructed = X_reconstructed.cpu().detach()

    # generate a grid with the reconstructed images and save them to file
    images = torchvision.utils.make_grid(X_reconstructed)
    filename = os.path.join(img_dir, 'image_at_epoch_{:04d}.png'.format(epoch))
    torchvision.utils.save_image(images, filename)


def generate_gif(img_dir, tbx_writer=None):

    anim_file = os.path.join(img_dir, img_dir.split('/')[-1] + '.gif')
    with imageio.get_writer(anim_file, mode='I', format='GIF', fps=10) as writer:
        filenames = glob.glob(os.path.join(img_dir, 'image*.png'))
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
    # if tbx_writer:
    #     gif_image = torch.from_numpy(np.array(Image.open(anim_file)))
    #     tbx_writer.add_image("animated_mnist_generation" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), gif_image)


if __name__=='__main__':
    root = os.path.abspath('..')
    # Load the parameters from json file
    args = generate_argument_parser()
    misc.tab_printer(args, "Argument Parameters")

    model_dir = os.path.join(root, args.model_dir)
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = misc.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # set batch size
    params.batch_size = 64

    # set run directory
    logs_dir = os.path.join(model_dir, 'runs')
    if args.run:
        log = args.run
        log_dir = os.path.join(logs_dir, args.run)
    else:
        log = sorted(glob.glob(os.path.join(logs_dir, '*')))[-1].split('/')[-1]
        log_dir = os.path.join(logs_dir, log)

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    misc.set_logger(os.path.join(root, args.model_dir, 'test.log'))

    # Create the input data pipeline
    logging.info("\nLoading the datasets...")

    # fetch dataloaders
    data_dir = os.path.join(root, args.data_dir)
    # fetch dataloaders
    if args.dataloader == 'mnist':
        import data_loaders.mnist_data_loader as data_loader

        dataloaders = data_loader.fetch_dataloader(types=['test'], data_dir=data_dir, download=False, params=params)

        # If output needs to be reshaped into an image

        # Fetch the model and loss function
        if args.model_dir.split('/')[-1] == 'info_vae':
            from model.info_VAE.information_maximizing_variational_autoencoder import VariationalAutoencoder

    elif args.dataloader == 'orl_face':
        import data_loaders.orl_data_loader as data_loader

        dataloaders = data_loader.fetch_dataloader(types=['test'], data_dir=data_dir, params=params)

        # fetch model
        from model.face_conv_VAE import VariationalAutoencoder

    test_dl = dataloaders['test']

    model = VariationalAutoencoder(params).cuda() if params.cuda else VariationalAutoencoder(params)
    # Reload weights from the saved file
    misc.load_checkpoint(os.path.join(log_dir, args.restore_file + log + '.pth.tar'), model)

    # Generate images
    if args.generate == 'gif':
        logging.info("\nGenerating gif\n")
        generate_images(model, model_dir, test_dl, params)
    elif args.generate == 'test':
        logging.info("\nReconstruct images form test set\n")
        reconstruct_images_form_test_set(model, test_dl, args.dataloader, params, log_dir)
    else:
        NotImplementedError("Generation not implemented, use gif or test.")

