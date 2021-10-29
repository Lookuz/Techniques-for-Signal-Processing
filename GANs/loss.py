import torch
from torch.autograd import grad, Variable

from wavegan import WaveGANDiscriminator, WaveGANGenerator

def compute_discriminator_loss(
    discriminator,
    x_real,
    x_generated,
    penalty_coefficient=10
):
    """
    Discriminator loss for GANs according to the WGAN-GP technique
    See more at: https://arxiv.org/abs/1704.00028
    """
    batch_size = x_real.shape[0]

    y_real = discriminator(x_real)
    y_generated = discriminator(x_generated)

    # Enforcing gradient penalty along straight lines between the two distributions
    epsilon = torch.FloatTensor(batch_size, 1, 1).uniform_(0, 1)

    x_hat = (1 - epsilon) * x_real.data + epsilon * x_generated.data
    x_hat = Variable(x_hat, requires_grad=True) # Activate gradients for gradient penalty

    y_hat = discriminator(x_hat) # Probability of sampled points of x_hat

    # Compute gradient penalty term
    gradients = grad(
        outputs=y_hat,
        inputs=x_hat,
        grad_outputs=torch.ones(y_hat.size()),
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