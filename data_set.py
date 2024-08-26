# Creado por Henar Larrinaga 2024
from torch.utils.data import Dataset
from adni_utils import load_adni_stack

class data_set(Dataset):
    def __init__(self, dbdir, imtype, which, downsample, final_size, lab, num_data, num_data2):
        super().__init__()
        self.dbdir = dbdir
        self.imtype = imtype
        self.which = which
        self.downsample = downsample
        self.final_size = final_size
        self.lab = lab
        self.num_data = num_data
        self.num_data2 = num_data2
        print("Initializing data_set...")
        self.stack, self.labels, self.imsize = self.load_data()

    def __len__(self):
        return len(self.labels)

    def load_data(self):
        stack, labels, imsize = load_adni_stack(
            dbdir=self.dbdir,
            imtype=self.imtype,
            which=self.which,
            downsample=self.downsample,
            final_size=self.final_size,
            lab=self.lab,
            num_data=self.num_data,
            num_data2=self.num_data2
        )
        return stack, labels, imsize 
    
    def __getitem__(self, index):
        sample = self.stack[index]
        label = self.labels[index]

        return sample, label
    
