# Creado por Henar Larrinaga 2024
import os
import torch
import modelos
from torch.utils.data import DataLoader
from data_set import data_set
from tqdm import tqdm
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image

dbdir='./'
#%% DATASET

dataset = data_set(dbdir=dbdir, imtype='gm', which='all', downsample=2, final_size=(81,81,81), lab='DX simple', num_data=None, num_data2=279)


def train_val_dataset(dataset, val_split=0.08):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets
datasets = train_val_dataset(dataset)


batch_size=2
train_loader = DataLoader(dataset=datasets['train'], batch_size=batch_size, shuffle=False)
val_loader = DataLoader(dataset=datasets['val'], batch_size=batch_size, shuffle=False)


# %% BUILD THE MODELS
latent_dim = 12
tabencoder = modelos.Conv3DVAEMDDEncoder(latent_dim=latent_dim).double()
tabdecoder = modelos.Conv3DVAEDecoder(latent_dim=latent_dim).double()
vae = modelos.GenVAE(encoder=tabencoder, decoder=tabdecoder)

device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
vae = vae.to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

# %% ITERACIONES

print('pasamos a iteraciones')
num_epochs = 300
PARAM_BETA = 1
PARAM_REDUCTION = 'mean'

train_loss = []
val_loss = []

save_directory = os.path.join("models", "nm_bce_bs2_latdim3_lr1e-4")
os.makedirs(save_directory, exist_ok=True)
save_epochs = [0, 1, 2, 3, 4, 9, 24, 49, 59, 74, 99, 124, 149, 199, 249, 299, 349, 399]


save_dir = "saved_images"
os.makedirs(save_dir, exist_ok=True)
class_labels = ['AD', 'CTL', 'MCI-S', 'MCI-C']
for label in class_labels:
    os.makedirs(os.path.join(save_dir, label), exist_ok=True)


for e in range(num_epochs):
    vae.train()
    train_epoch = [] 

    for ix, data_batch in enumerate(tqdm(train_loader)):  
        optimizer.zero_grad()
        
        image_batch, label_batch = data_batch
        image_batch = image_batch.double()
        image_batch = image_batch.to(device)
        image_batch = image_batch.unsqueeze(1)
        image_batch = torch.clamp(image_batch, 0, 1)

        # forward pass:
        z, z_mean, z_logvar, x_recon = vae(image_batch)

        loss, recon_loss, kl_loss = vae.loss_function(image_batch, x_recon, z_mean, z_logvar, beta=PARAM_BETA, reduction=PARAM_REDUCTION)
        loss.backward()
        optimizer.step()
        train_epoch.append(loss.item())

    train_loss.append(sum(train_epoch)/len(train_epoch))
    print(f'E: {e}, train loss: {train_loss[-1]}')
    
    val_epoch = []
    
    for ix_val, data_batch_val in enumerate(tqdm(val_loader)): 
        image_batch_val, label_batch_val = data_batch_val
        image_batch_val = image_batch_val.double()
        image_batch_val = image_batch_val.to(device)
        image_batch_val = image_batch_val.unsqueeze(1)
        image_batch_val = torch.clamp(image_batch_val, 0, 1)

        # forward pass:
        z_val, z_mean_val, z_logvar_val, x_recon_val = vae(image_batch_val)

        loss_val, recon_loss_val, kl_loss_val = vae.loss_function(image_batch_val, x_recon_val, z_mean_val, z_logvar_val, beta=PARAM_BETA, reduction=PARAM_REDUCTION)
        val_epoch.append(loss_val.item())

    val_loss.append(sum(val_epoch)/len(val_epoch))
    print(f'E: {e}, validation loss: {val_loss[-1]}')
    
    if e in save_epochs:
        model_filename = f"trading_model_epoch_{e}.pt"
        save_path = os.path.join(save_directory, model_filename)
        torch.save(vae, save_path)
 
        # Save one image per class
        for i, label in enumerate(label_batch): 
            label_dir = os.path.join(save_dir, label)
            original_image = image_batch[i, 0, :, :, :]
            save_image(original_image[:, :, 40], os.path.join(label_dir, f"{label}_epoch{e}_{i}train_original.png"))
            recon_image = x_recon[i, 0, :, :, :]
            save_image(recon_image[:, :, 40], os.path.join(label_dir, f"{label}_epoch{e}_{i}train_recons.png"))

        for i, label in enumerate(label_batch_val):  
            label_dir = os.path.join(save_dir, label)
            original_image = image_batch_val[i, 0, :, :, :]
            save_image(original_image[:, :, 40], os.path.join(label_dir, f"{label}_epoch{e}_{i}val_original.png"))
            recon_image = x_recon_val[i, 0, :, :, :]
            save_image(recon_image[:, :, 40], os.path.join(label_dir, f"{label}_epoch{e}_{i}val_recons.png"))

print(f'train_loss={train_loss}')
print(f'val_loss={val_loss}')
