import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import grad, Variable
from tqdm import tqdm

from wavegan import WaveGANDiscriminator, WaveGANGenerator
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_discriminator_loss(
    discriminator,
    x_real,
    x_generated,
    penalty_coefficient=1,
    device=device
):
    """
    Discriminator loss for GANs according to the WGAN-GP technique
    See more at: https://arxiv.org/abs/1704.00028
    """
    batch_size = x_real.shape[0]

    y_real = discriminator(x_real)
    y_generated = discriminator(x_generated)

    # Enforcing gradient penalty along straight lines between the two distributions
    epsilon = torch.FloatTensor(batch_size, 1, 1).uniform_(0, 1).to(device)

    x_hat = (1 - epsilon) * x_real + epsilon * x_generated
    x_hat = Variable(x_hat, requires_grad=True) # Activate gradients for gradient penalty

    y_hat = discriminator(x_hat) # Probability of sampled points of x_hat

    # Compute gradient penalty term
    gradients = grad(
        outputs=y_hat,
        inputs=x_hat,
        grad_outputs=torch.ones(y_hat.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradient_penalty = penalty_coefficient * (
        (gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2
    )
    gradient_penalty = gradient_penalty.mean()

    # Overall loss = Wasserstein distance + gradient penalty
    loss = (y_generated.mean() - y_real.mean()) + gradient_penalty

    return loss

def train(
    x,
    generator,
    discriminator,
    noise_dim=100,
    # Optimizers for generator and discriminator netorks
    optimizer_g=None,
    optimizer_d=None,
    # Optimizer parameters
    penalty_coefficient=10,
    n_critic=5,
    alpha=0.0001,
    beta_1=0, beta_2=0.9,
    # Training parameters
    epochs=100,
    batch_size=64,
    verbose=True
):
    """
    Runs the training routine for the GAN according to the WGAN-GP strategy

    Args:
        x: Real data to be used as input into the generator
        generator, discriminator: Generator and discriminator networks respectively
        optimizer_g, optimizer_d: Optimizers to be used for the generator and discriminator
        penalty_coefficient(lambda), alpha, beta_1, beta_2: Parameters for the default optimizer(Adam)
    """
    if verbose:
        if torch.cuda.is_available():
            print('CUDA device detected. Utilising GPU.')
            device = torch.device('cuda')
        else:
            print('CUDA device not detected. Switching to CPU.')
            device = torch.device('cpu')


    if optimizer_g is None:
        optimizer_g = optim.Adam(
            generator.parameters(),
            lr=alpha, betas=(beta_1, beta_2)
        )
    if optimizer_d is None:
        optimizer_d = optim.Adam(
            discriminator.parameters(),
            lr=alpha, betas=(beta_1, beta_2)
        )
    
    generator.train()
    discriminator.train()

    if verbose:
        print("Beginning training...")

    for i in tqdm(range(epochs)):
        # Train only the generator
        set_trainable_gradients(generator, False)
        set_trainable_gradients(discriminator, True)
        generator.eval()
        discriminator.train()

        # Split data into random batches
        x_dataloader = iter(DataLoader(
            x, batch_size=batch_size,
            shuffle=True
        ))

        # Compute discriminator loss
        for j in range(n_critic):
            # Sample random batch
            x_real = next(x_dataloader)

            # Sample generation noise samples and generated outputs
            noise_vectors = sample_noise(batch_size, dim=noise_dim, device=device)
            x_generated = generator(noise_vectors)

            # Compute discriminator loss
            generator.zero_grad()
            optimizer_g.zero_grad()
            discriminator.zero_grad()
            optimizer_d.zero_grad()

            # Discriminator loss using Wasserstein distance with gradient penalty
            loss = compute_discriminator_loss(
                discriminator, x_real, x_generated, penalty_coefficient=penalty_coefficient
            )
            loss.backward()
            optimizer_d.step()

        # Compute generator loss
        set_trainable_gradients(generator, True)
        set_trainable_gradients(discriminator, False)
        generator.train()
        discriminator.eval()

        generator.zero_grad()
        optimizer_g.zero_grad()
        discriminator.zero_grad()
        optimizer_d.zero_grad()

        # Loss function using negative of discriminator output
        noise_vectors = sample_noise(batch_size, dim=noise_dim, device=device)
        x_generated = generator(noise_vectors)

        discriminator_output = discriminator(x_generated)
        loss = - discriminator_output.mean()

        loss.backward()
        optimizer_g.step()

    if verbose:
        print("Training completed")
    
    return generator, discriminator
