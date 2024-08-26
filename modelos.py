import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class VAEEncoder(nn.Module):
    """Empty class for generic VAE Encoder. 
    
    Args:
        latent_dim (int, optional): Dimension of the latent space. Defaults to 20.

    Methods
    -------
    reparameterize(mu, logvar):
        Performs the reparameterization trick

    divergence_loss(mu, logvar, reduction='sum'):
        Computes the Kullback-Leibler divergence between the latent distribution (mu,logvar) and N(0,1)
    """

    def __init__(self, latent_dim:int=20, kws_loss:dict={'reduction':'sum'}) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.kws_loss = kws_loss

    def forward(self, x:Tensor) -> Tensor:
        return x

    def reparameterize(self, mu:Tensor, logvar:Tensor) -> Tensor:
        """Performs the reparameterization trick

        Args:
            mu (Tensor): tensor of means
            logvar (Tensor): tensor of log(variance) 

        Returns:
            Tensor: sampled tensor at the Z layer
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def divergence_loss(self, mu:Tensor, logvar:Tensor) -> Tensor:
        """Computes the Kullback-Leibler divergence between the latent distribution (mu,logvar) and N(0,1)

        Args:
            mu (Tensor): tensor of means
            logvar (Tensor): tensor of log(variance) 
            reduction (str, optional): Aggregation function for KL loss. Defaults to 'sum'.

        Returns:
            Tensor: KL divergence
        """
        kl_batch =  -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        if self.kws_loss['reduction']=='sum':
            return kl_batch.sum()
        else:
            return kl_batch.mean()

class MDDEncoder(VAEEncoder):
    def _compute_kernel(self, x:Tensor, y:Tensor) -> Tensor:
        """Compute gaussian kernel for MMD distance metrics.

        Args:
            x (Tensor): _description_
            y (Tensor): _description_

        Returns:
            Tensor: kernel 
        """
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]

        tiled_x = x.view(x_size,1,dim).repeat(1, y_size,1)
        tiled_y = y.view(1,y_size,dim).repeat(x_size, 1,1)

        return torch.exp(-torch.mean((tiled_x - tiled_y)**2,dim=2)/dim*1.0)

    def compute_mmd(self, x:Tensor, y:Tensor) -> Tensor:
        """Calcula el MDD between x and y. 

        Args:
            x (Tensor): _description_
            y (Tensor): _description_
            reduction (str, optional): _description_. Defaults to 'mean'.

        Returns:
            Tensor: _description_
        """
        x_kernel = self._compute_kernel(x, x)
        y_kernel = self._compute_kernel(y, y)
        xy_kernel = self._compute_kernel(x, y)
        if self.kws_loss['reduction']=='sum':
            return torch.sum(x_kernel) + torch.sum(y_kernel) - 2*torch.sum(xy_kernel)
        else:
            return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)
        
    
    def divergence_loss(self, mu:Tensor, logvar:Tensor) -> Tensor:
        """Computes the Maximum Mean Discrepancy (arXiv:0805.2368) between the sampled latent distribution (mu=z, logvar=None) and N(0,1)

        Args:
            mu (Tensor): tensor of means
            logvar (Tensor): tensor of log(variance) 
            reduction (str, optional): Aggregation function for KL loss. Defaults to 'sum'.

        Returns:
            Tensor: KL divergence
        """
        true_samples = Variable(
                        torch.randn(len(mu), self.latent_dim),
                        requires_grad=False
                    ).to(torch.device(mu.get_device()))
        return self.compute_mmd(true_samples, mu)

class VAEDecoder(nn.Module):
    """Empty class for generic VAE Decoder. 
    
    Args:
        latent_dim (int, optional): Dimension of the latent space. Defaults to 20.
    
    Methods
    -------
    recon_loss(predictions, targets, reduction='sum'):
        Computes the VAE reconstruction loss assuming gaussian distribution of the data.
    """

    def __init__(self, latent_dim:int=20, recon_function:callable=F.mse_loss, kws_loss:dict={'reduction':'sum'}) -> None: 
        super().__init__()
        self.latent_dim = latent_dim
        self.recon_function = recon_function
        self.kws_loss = kws_loss

    def forward(self, x:Tensor) -> Tensor:
        return x
    
    def recon_loss(self, targets:Tensor, predictions:Tensor) -> Tensor:
        """Computes the VAE reconstruction loss assuming gaussian distribution of the data.

        Args:
            targets (Tensor): Target (original) values of the input data
            predictions (Tensor): Predicted values for the input sample
            reduction (str, optional): Aggregation function for the reconstruction loss. Defaults to 'sum'.

        Returns:
            Tensor: Reconstruction loss
        """
        # From arXiv calibrated decoder: arXiv:2006.13202v3
        # D is the dimensionality of x. 
        r_loss = self.recon_function(predictions, targets, **self.kws_loss)
        # torch.pow(predictions-targets, 2).mean(dim=(1,2,3,4)) #+ D * self.logsigma
        return r_loss


class Conv3DVAEEncoder(VAEEncoder):
    """Implementation of 3D Convolutional VAE Decoder. 

    Args:
        latent_dim (int, optional): Dimension of the latent space. Defaults to 20.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Encoder
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1) # 294 912 pesos
        self.fc1 = nn.Linear(256 * 6 * 6 * 6, 512) # 37 748 736 pesos
        self.fc2_mean = nn.Linear(512, self.latent_dim)
        self.fc2_logvar = nn.Linear(512, self.latent_dim)

    def forward(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Encode
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 256 * 6 * 6 * 6)
        x = F.relu(self.fc1(x))
        z_mean = self.fc2_mean(x)
        z_logvar = self.fc2_logvar(x)
        z = self.reparameterize(z_mean, z_logvar)
        return z, z_mean, z_logvar
    
class Conv3DVAEMDDEncoder(MDDEncoder):
    """Implementation of 3D Convolutional VAE Decoder. 

    Args:
        latent_dim (int, optional): Dimension of the latent space. Defaults to 20.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Encoder
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1) # 294 912 pesos
        self.fc1 = nn.Linear(256 * 6 * 6 * 6, 512) # 37 748 736 pesos
        self.fc2_mean = nn.Linear(512, self.latent_dim)
        self.fc2_logvar = nn.Linear(512, self.latent_dim)

    def forward(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Encode
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 256 * 6 * 6 * 6)
        x = F.relu(self.fc1(x))
        z_mean = self.fc2_mean(x)
        z_logvar = self.fc2_logvar(x)
        z = self.reparameterize(z_mean, z_logvar)
        return z, z_mean, z_logvar
    
class Conv3DVAEDecoder(VAEDecoder):
    """Implementation of 3D Convolutional VAE Decoder. 

    Args:
        latent_dim (int, optional): Dimension of the latent space. Defaults to 20.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(Conv3DVAEDecoder, self).__init__(*args, **kwargs)

        # Decoder
        self.fc1 = nn.Linear(self.latent_dim, 256 * 6 * 6 * 6)
        self.conv1 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.conv2 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.conv3 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.conv4 = nn.ConvTranspose3d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=0)

    def forward(self, x:Tensor) -> Tensor:
        # Decode
        x = F.relu(self.fc1(x))
        x = x.view(-1, 256, 6, 6, 6)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.sigmoid(self.conv4(x))
        return x


class GenVAE(nn.Module):
    """Generic implementation for a VAE, involving a encoder and decoder, and necessary fuctions. 

    Args:
        encoder (VAEEncoder): Encoder architecture to be used. 
        decoder (VAEDecoder): Decoder architecture to be used. 
    """
    def __init__(self, encoder:VAEEncoder, decoder:VAEDecoder):
        super().__init__()
        assert encoder.latent_dim == decoder.latent_dim, "latent_dim of encoder and decoder must be equal"

        self.encode = encoder
        self.decode = decoder

    def forward(self, x:Tensor) ->  tuple[Tensor, Tensor, Tensor, Tensor]:
        # Encode
        z, z_mean, z_logvar = self.encode(x)

        # Decode
        x_recon = self.decode(z)

        return z, z_mean, z_logvar, x_recon
    
    def loss_function(
            self, 
            x:Tensor, 
            x_recon:Tensor, 
            z_mean:Tensor, 
            z_logvar:Tensor, 
            beta:float=1., 
            reduction:str='sum'
            )  -> tuple[Tensor, Tensor, Tensor]:
        """ Generic loss function. 

        Args:
            x (Tensor): _description_
            x_recon (Tensor): _description_
            z_mean (Tensor): _description_
            z_logvar (Tensor): _description_
            beta (float, optional): _description_. Defaults to 1..
            reduction (str, optional): _description_. Defaults to 'sum'.

        Returns:
            tuple[Tensor, Tensor, Tensor]: _description_
        """
        divergence_loss = self.encode.divergence_loss(z_mean, z_logvar)
        recon_loss = self.decode.recon_loss(x, x_recon)
        return recon_loss + beta*divergence_loss, recon_loss, divergence_loss

