# Creado por Henar Larrinaga 2024
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from data_set import data_set
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import cross_validate, cross_val_predict, RepeatedStratifiedKFold
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

dbdir='./'

#%% DATASET

dataset = data_set(dbdir=dbdir, imtype='gm', which='all', downsample=2, final_size=(81,81,81), lab='DX simple', num_data=279, num_data2=0)
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# %% LOAD MODELS
save_directory = os.path.join("models", "bce_bs2_latdim12_lr1e-4") # Definir la carpeta donde se guardaron los modelos
model_filename = "trading_model_epoch_49.pt"                       # Nombre del archivo del modelo que queremos cargar
model_path = os.path.join(save_directory, model_filename)          # Crear la ruta completa al archivo del modelo

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

vae = torch.load(model_path, map_location=device) # Cargar el modelo
vae = vae.to(device)


# %% TEST

save_dir = "saved_images"
os.makedirs(save_dir, exist_ok=True)
class_labels = ['AD', 'CTL', 'MCI-S', 'MCI-C']
for label in class_labels:
    os.makedirs(os.path.join(save_dir, label), exist_ok=True)


PARAM_BETA = 1
PARAM_REDUCTION = 'mean'
all_loss = []
labels = []
all_z = []


for ix, data_batch in enumerate(tqdm(test_loader)):  
    image_batch, label_batch = data_batch
    image_batch = image_batch.double()
    image_batch = image_batch.to(device)
    image_batch = image_batch.unsqueeze(1)
    image_batch = torch.clamp(image_batch, 0, 1)
    
    z, z_mean, z_logvar, x_recon = vae(image_batch) # forward pass:
    labels.extend(label_batch)
    all_z.append(z.detach().cpu().numpy())

    z_mean = z_mean.to(device)
    z_logvar = z_logvar.to(device)
    x_recon = x_recon.to(device)

    loss, recon_loss, kl_loss = vae.loss_function(image_batch, x_recon, z_mean, z_logvar, beta=PARAM_BETA, reduction=PARAM_REDUCTION)
    all_loss.append(loss.item())
    
    for i, label in enumerate(label_batch):
        label_dir = os.path.join(save_dir, label)
        
        # Save original image
        original_image = image_batch[i, 0, :, :, :]
        save_image(original_image[:, :, 40], os.path.join(label_dir, f"{ix}_{i}_original.png"))

        # Save reconstructed image
        recon_image = x_recon[i, 0, :, :, :]
        save_image(recon_image[:, :, 40], os.path.join(label_dir, f"{ix}_{i}_recons.png"))
    

print(f'test loss: {all_loss[-1]}]')
print(labels)
print(f'losses = {all_loss}')

all_z = np.vstack(all_z)


#%% cross validation
model = svm.LinearSVC()
results = cross_validate(model, all_z, y=labels, scoring=['accuracy', 'balanced_accuracy'], cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10), n_jobs=5, return_estimator=True)
mean_accuracy = results['test_accuracy'].mean() 
mean_balanced_accuracy = results['test_balanced_accuracy'].mean()
print(f'Mean Accuracy: {mean_accuracy}')
print(f'Mean balanced accuracy: {mean_balanced_accuracy}')


#%% Confusion matrix
labels = np.array(labels)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10) # Perform repeated stratified k-fold cross-validation
y_true = []
y_pred = []
for train_index, test_index in cv.split(all_z, labels):
    X_train, X_test = all_z[train_index], all_z[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    y_true.extend(y_test)
    y_pred.extend(predictions)
y_true = np.array(y_true) 
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred) # Compute the confusion matrix

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues): # Define the function to plot confusion matrix
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
classes = np.unique(labels) # Get class names from labels

plot_confusion_matrix(cm, classes) # Plot confusion matrix
plt.show()

print(len(y_pred))
print(len(labels)) # hace 10 iteraciones, entonces hay 10 veces más números


#%% Confusion matrix en porcentajes
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

def plot_confusion_matrix_per(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f'  # Show two decimal places
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt) + '%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix_per(cm_percentage, classes)
plt.show()

# %% SCATTER PLOT
dir2 = 'scatter_1dataset'
os.makedirs(dir2, exist_ok=True)
label_to_color = {
    'CTL': 'blue',
    'AD': 'red',
    'MCI-C': 'green',
    'MCI-S': 'orange'
}
colors = [label_to_color[label] for label in labels]
last = 12
plt.figure(5)
plt.scatter(all_z[:,last-1], all_z[:,0], c=colors, cmap='viridis')
filepath = os.path.join(dir2, f'scatter_plot_last.png')
plt.savefig(filepath)
plt.close()  # Cerrar la figura para liberar memoria
for i in range(len(all_z)-1):
    plt.figure(i)
    plt.scatter(all_z[:,i], all_z[:,i+1], c=colors, cmap='viridis')

    filepath = os.path.join(dir2, f'scatter_plot_{i+1}.png')
    plt.savefig(filepath)
    plt.close()  # Cerrar la figura para liberar memoria


# %% GENERAR IMAGENES

ddata = 12 
Nsamp = 10
znom = (torch.linspace(-100, 50, Nsamp), ) * ddata
outputs = torch.meshgrid(*znom)
zsamp = torch.stack([el.flatten() for el in outputs], axis=0).T
zsamp_fijo = zsamp[(zsamp[:,0]==2)&(zsamp[:,1]==-5)]

batch_size = 1
outcomes = []
vae = vae.float()
with torch.no_grad():
    vae.eval()
    for i in range(0, zsamp_fijo.size(0), batch_size):
        batch = zsamp_fijo[i:i+batch_size].to(device).float()
        outcomes.append(vae.decode(batch).cpu())

# See outcomes
slice_index = 40                  
n_images = min(len(outcomes), 5)  

fig, axes = plt.subplots(3, n_images, figsize=(15, 15))

for i in range(n_images):
    tensor = outcomes[i][0].squeeze()  
    
    # Vista Axial
    ax = axes[0, i]
    ax.imshow(tensor[:, :, slice_index].numpy(), cmap='gray')
    ax.set_title(f'Vista Axial {i+1}')
    ax.axis('off')

    # Vista Sagital
    ax = axes[1, i]
    ax.imshow(tensor[slice_index, :, :].numpy(), cmap='gray')
    ax.set_title(f'Vista Sagital {i+1}')
    ax.axis('off')

    # Vista Coronal (con rotación de 90 grados hacia la izquierda)
    ax = axes[2, i]
    img_rotated = np.rot90(tensor[:, slice_index, :].numpy(), k=1)
    ax.imshow(img_rotated, cmap='gray')
    ax.set_title(f'Vista Coronal {i+1}')
    ax.axis('off')

plt.tight_layout()
plt.show()

#%% Generar imágenes dimlat12 

tensor = torch.zeros((10, 12))
tensor[:,0] = tensor[:,1] = tensor[:,3] = tensor[:,10] = tensor[:,11] = 0
tensor[:,2] = tensor[:,9] = -25
tensor[:,4] = tensor[:,5] = tensor[:,7] = tensor[:,8] = 50
tensor[:,6] = torch.tensor(np.linspace(-100, 50, 10))
zsamp_fijo = tensor
print(zsamp_fijo)

# Procesamiento por lotes
batch_size = 1
outcomes = []
vae = vae.float()
with torch.no_grad():
    vae.eval()
    for i in range(0, zsamp_fijo.size(0), batch_size):
        batch = zsamp_fijo[i:i+batch_size].to(device).float()
        outcomes.append(vae.decode(batch).cpu())

# See outcomes
slice_index = 40  
n_images = min(len(outcomes), 5)  

fig, axes = plt.subplots(3, n_images, figsize=(15, 15))

for i in range(n_images):
    tensor = outcomes[i][0].squeeze()  
    
    # Vista Axial
    ax = axes[0, i]
    ax.imshow(tensor[:, :, slice_index].numpy(), cmap='gray')
    ax.set_title(f'Vista Axial {i+1}')
    ax.axis('off')

    # Vista Sagital
    ax = axes[1, i]
    ax.imshow(tensor[slice_index, :, :].numpy(), cmap='gray')
    ax.set_title(f'Vista Sagital {i+1}')
    ax.axis('off')

    # Vista Coronal (con rotación de 90 grados hacia la izquierda)
    ax = axes[2, i]
    img_rotated = np.rot90(tensor[:, slice_index, :].numpy(), k=1)
    ax.imshow(img_rotated, cmap='gray')
    ax.set_title(f'Vista Coronal {i+1}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# %%
