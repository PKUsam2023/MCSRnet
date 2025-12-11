import os
import torch
from os.path import join
from torch.utils.data import Dataset
from .config import CFG
from pymatgen.core import Structure
import numpy as np
import torch.nn.functional as F

class CXSPDataset(Dataset):
    def __init__(self, mode, transforms=None):
        super().__init__()

        self.transforms = transforms

        # Select dataset path based on mode
        if mode == "train" or mode == "train_apply":
            self.path = CFG.train_path
        elif mode == "test":
            self.path = CFG.test_path
        elif mode == "valid":
            self.path = CFG.val_path

        # Paths for CIF and XRD files
        self.cif_path = join(self.path, "cif")
        self.xrd_path = join(self.path, "xrd")

        # List all CIF filenames
        self.path_name_list = sorted(os.listdir(self.cif_path))

    def __getitem__(self, index):
        # Load CIF structure
        s = Structure.from_file(join(self.cif_path, self.path_name_list[index]))

        # Extract occupancy ratios, atomic positions, and lattice vectors
        cif_occupy, cif_atom, cif_lattice = self.get_occupancy_ratios(s)

        # Load corresponding XRD data and normalize
        xrd = torch.load(join(self.xrd_path, self.path_name_list[index].replace("cif", "pt")))
        xrd = F.normalize(xrd, p=2, dim=0)

        # Use filename as tokenizer input
        tokenizer = self.path_name_list[index]

        # Move tensors to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cif_lattice = cif_lattice.to(device)
        cif_atom = cif_atom.to(device)
        cif_occupy = cif_occupy.to(device)
        xrd = xrd.to(device)

        # Return all model inputs
        return cif_lattice, cif_atom, xrd, cif_occupy, tokenizer

    def __len__(self):
        # Number of samples in the dataset
        return len(self.path_name_list)

    def get_occupancy_ratios(self, struct):
        occupancy_ratios = []

        # Extract fractional coordinates of all atoms (Nx3 array)
        frac_coords = np.array([site.frac_coords for site in struct])

        for site in struct:
            # Get species and their occupancies at the site
            species = site.species
            species_and_occu = list(species.items())

            # If only one species, use its occupancy
            if len(species_and_occu) == 1:
                element, occupancy = species_and_occu[0]
                occupancy_ratios.append(occupancy)
            else:
                # If mixed species, take the one with the highest occupancy
                species_and_occu_sorted = sorted(
                    species_and_occu, key=lambda x: x[1], reverse=True
                )
                element1, occ1 = species_and_occu_sorted[0]
                occupancy_ratios.append(occ1)

        # Convert occupancy list to tensor
        cif_occupy = torch.tensor(occupancy_ratios)

        # Flatten fractional coordinates to length 15
        cif_atom = torch.tensor(frac_coords).reshape(15)

        # Extract lattice matrix (3x3) and flatten to length 9
        lattice = torch.tensor(struct.lattice.matrix)
        cif_lattice = lattice.reshape(9)

        # Normalize atomic coordinate vector
        cif_atom = F.normalize(cif_atom, p=2, dim=0)

        return cif_occupy, cif_atom, cif_lattice
