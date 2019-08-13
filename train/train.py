"""Train the model"""
import logging
import os
import torch
import sys
from datetime import datetime
import torch.optim as optim
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from torchcontrib.optim import SWA

# add top-level in order to access utils folder
sys.path.append("..")
from utils.parsers import train_argument_parser
from utils.misc import create_dir, create_log_dir, load_checkpoint, save_checkpoint, tab_printer, Params, set_logger
from generate.generate_images import generate_and_save_images, generate_gif


def train(model, dataloader, dl_type, optimizer, loss_fn, params, model_dir, use_swa, restore_file=None):
    """
    Train the model on `num_steps` batches

    Parameters
    ----------
    model
    dataloader
    dl_type
    optimizer
    loss_fn
    params
    model_dir
    reshape
    restore_file

    Returns
    -------

    """

    random_vector_for_generation = torch.randn(torch.Size([params.num_examples_to_generate, params.latent_dim])).cuda()

    logging.info("\nTraining started.\n")
    # Add tensorboardX SummeryWriter to log training, logs will be save in model_dir directory
    run = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = create_log_dir(model_dir, run)
    img_dir = os.path.join(log_dir, 'generated_images')
    create_dir(img_dir)
    with SummaryWriter(log_dir) as writer:
        # set model to training mode
        model.train()

        # reload weights from restore_file if specified
        if restore_file is not None:
            restore_path = os.path.join(model_dir, args.restore_file + '.pth.tar')
            logging.info("Restoring parameters from {}".format(restore_path))
            load_checkpoint(restore_path, model, optimizer)

        # number of iterations
        iterations = 0
        # Use tqdm progress bar for number of epochs
        for epoch in tqdm(range(params.num_epochs), desc="Epochs: ", leave=True):
            # Track the progress of the training batches
            training_progressor = trange(len(dataloader), desc="Loss")
            for i in training_progressor:
                iterations += 1
                # Fetch next batch of training samples
                if dl_type == 'orl_face':
                    train_batch = next(iter(dataloader))
                else:
                    train_batch, _ = next(iter(dataloader))

                true_samples = torch.randn(params.batch_size, params.latent_dim)

                # move to GPU if available
                if params.cuda:
                    train_batch = train_batch.cuda()
                    true_samples = true_samples.cuda()

                X_reconstructed, z = model(train_batch)
                losses = loss_fn(train_batch, X_reconstructed, true_samples, z)
                loss = losses['loss']

                # clear previous gradients, compute gradients of all variables wrt loss
                optimizer.zero_grad()
                loss.backward()

                # performs updates using calculated gradients
                optimizer.step()

                # Evaluate model parameters only once in a while
                if (i + 1) % params.save_summary_steps == 0:
                    # Log values and gradients of the model parameters (histogram summary)
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram(tag, value.cpu().data.numpy(), iterations)
                        writer.add_histogram(tag + '/grad', value.grad.cpu().data.numpy(), iterations)

                # Compute the loss for each iteration
                summary_batch = losses
                # log loss and/or other metrics to the writer
                for tag, value in summary_batch.items():
                    writer.add_scalar(tag, value.item(), iterations)

                # update the average loss
                training_progressor.set_description("VAE (Loss=%g)" % round(loss.item(), 4))

            # generate images for gif
            if epoch % 1 == 0:
                generate_and_save_images(model, epoch, random_vector_for_generation, img_dir)

            # Save weights
            if epoch % 10 == 0:
                save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': model.state_dict(),
                                 'optim_dict': optimizer.state_dict()},
                                is_best=True,
                                checkpoint=log_dir,
                                datetime=run)

        logging.info("\n\nTraining Completed.\n\n")
        if use_swa:
            optimizer.swap_swa_sgd()

        logging.info("Creating gif of images generated with gaussian latent vectors.\n")
        generate_gif(img_dir, writer)


if __name__ == '__main__':
    # add top-level as root
    root = os.path.abspath('..')

    # arguments for the training script
    args = train_argument_parser()

    # Load the parameters from json file
    model_dir = os.path.join(root, args.model_dir)
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    set_logger(os.path.join(root, args.model_dir, 'train.log'))
    # print out the arguments in a nice table
    tab_printer(args, "Argument Parameters")
    tab_printer(params, "Hyperparameters")
    # Create the input data pipeline
    logging.info("\nLoading the datasets...")

    data_dir = os.path.join(root, args.data_dir)

    # fetch dataloaders
    if args.dataloader == 'mnist':
        import data_loaders.mnist_data_loader as data_loader
        dataloaders = data_loader.fetch_dataloader(types=['train'], data_dir=data_dir, download=False, params=params)

        # Fetch the model and loss function
        if args.model_dir.split('/')[-1] == 'info_vae':
            from model.info_VAE.information_maximizing_variational_autoencoder import VariationalAutoencoder

    elif args.dataloader == 'orl_face':
        import data_loaders.orl_data_loader as data_loader
        dataloaders = data_loader.fetch_dataloader(types=['train'], data_dir=data_dir, params=params)

        # fetch model
        reshape = False
        from model.face_conv_VAE import VariationalAutoencoder

    train_dl = dataloaders['train']

    # Fetch the loss
    from model.loss_functions.mmd_loss import loss_function
    
    loss_fn = loss_function

    # Load the VAE model
    model = VariationalAutoencoder(params).cuda() if params.cuda else VariationalAutoencoder(params)

    if args.swa:
        # Use the Adam optimizer
        base_optimizer = optim.Adam(model.parameters(),
                               lr=1e-3,
                               eps=params.eps,
                               betas=(params.betas[0], params.betas[1]),
                               weight_decay=params.weight_decay)
        optimizer = SWA(base_optimizer, swa_start=10, swa_freq=5, swa_lr=1e-3)

    else:
        # Use the Adam optimizer
        optimizer = optim.Adam(model.parameters(),
                               lr=params.learning_rate,
                               eps=params.eps,
                               betas=(params.betas[0], params.betas[1]),
                               weight_decay=params.weight_decay)


    # Train the model
    train(model, train_dl, args.dataloader, optimizer, loss_fn, params, model_dir, args.swa, args.restore_file)
