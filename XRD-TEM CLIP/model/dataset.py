import os
import torch
from os.path import join
from torch.utils.data import Dataset
from .config import CFG
from pymatgen.core import Structure
import numpy as np
import torch.nn.functional as F
import cv2 as cv


class XTSPDataset(Dataset):
    def __init__(self, mode, transforms=None):
        super().__init__()

        # Select dataset path based on mode
        self.transforms = transforms
        if mode == "train" or mode == "train_apply":
            self.path = CFG.train_path
        elif mode == "test":
            self.path = CFG.test_path
        elif mode == "valid":
            self.path = CFG.val_path

        # Sub-folder paths
        self.cif_path = join(self.path, "cif")
        self.xrd_path = join(self.path, "xrd")
        self.tem_path = join(self.path, "tem")

        # List all sample names (limit to 128)
        self.path_name_list = sorted(os.listdir(self.cif_path))[:128]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transf = transforms.ToTensor()

        # TEM view list
        self.temname = ['100.png', '010.png', '001.png',
                        '110.png', '101.png', '011.png', '111.png']

        # Data buffers
        self.tem_data = []
        self.occupy_data = []
        self.atom_data = []
        self.lattice_data = []
        self.xrd_data = []

        # Preload CIF/XRD information
        for file_name in self.path_name_list:
            cif_file_path = join(self.cif_path, file_name)

            # Load CIF-related features
            cif = self.get_occupancy_ratios(cif_file_path)
            self.occupy_data.append(cif[0])
            self.atom_data.append(cif[1])
            self.lattice_data.append(cif[2])

            # Load and normalize XRD features
            xrd = torch.load(join(self.xrd_path, file_name.replace("cif", "pt")))
            xrd = F.normalize(xrd, p=2, dim=0).to(device)
            self.xrd_data.append(xrd)

    def __getitem__(self, index):

        # Retrieve CIF features
        tokenizer = self.path_name_list[index]
        cif_occupy = self.occupy_data[index]
        cif_atom = self.atom_data[index]
        cif_lattice = self.lattice_data[index]
        xrd = self.xrd_data[index]

        # Load 7-view TEM images
        tmp_list = []
        for p in self.temname:
            path_tem = join(self.tem_path, self.path_name_list[index].replace(".cif", ""), p)
            img = cv.imread(path_tem, cv.IMREAD_GRAYSCALE)
            img_tensor = self.transf(img)
            tmp_list.append(img_tensor)

        # Concatenate 7 images into a 7-channel tensor
        tem = torch.cat(tmp_list, dim=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tem = tem.to(device)

        return cif_lattice, cif_atom, xrd, cif_occupy, tem, tokenizer

    def __len__(self):
        return len(self.path_name_list)

    def get_occupancy_ratios(self, file_path):
        # Load CIF file
        struct = Structure.from_file(file_path)
        occupancy_ratios = []

        # Extract fractional coordinates
        frac_coords = np.array([site.frac_coords for site in struct])

        # Extract highest occupancy element for each site
        for site in struct:
            species_and_occu = list(site.species.items())
            if len(species_and_occu) == 1:
                occupancy_ratios.append(species_and_occu[0][1])
            else:
                species_sorted = sorted(species_and_occu, key=lambda x: x[1], reverse=True)
                occupancy_ratios.append(species_sorted[0][1])

        # Tensors
        cif_occupy = torch.tensor(occupancy_ratios)
        cif_atom = torch.tensor(frac_coords).reshape(15)
        lattice = torch.tensor(struct.lattice.matrix).reshape(9)

        # Normalize atom coordinates
        cif_atom = F.normalize(cif_atom, p=2, dim=0)

        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cif_lattice = lattice.to(device)
        cif_atom = cif_atom.to(device)
        cif_occupy = cif_occupy.to(device)

        return cif_occupy, cif_atom, cif_lattice

    def get_tem(self, file_path):
        # Load 7-view TEM images
        tmp_list = []
        for p in self.temname:
            path_tem = join(file_path, p)
            img = cv.imread(path_tem, cv.IMREAD_GRAYSCALE)
            img_tensor = self.transf(img)
            tmp_list.append(img_tensor)

        # Stack all views
        tem = torch.cat(tmp_list, dim=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tem = tem.to(device)

        return tem
