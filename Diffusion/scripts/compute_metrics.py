from collections import Counter
import argparse
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty
import time
import pymatgen
from pymatgen.analysis.diffraction import xrd
import random
import math
import torch
from os.path import join
import sys
sys.path.append('.')
from eval_utils import (
    smact_validity, structure_validity, CompScaler, get_fp_pdist,
    load_config, load_data, get_crystals_list, prop_model_eval, compute_cov)

element_to_atomic_number = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19,
    'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28,
    'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37,
    'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46,
    'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56,
    'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
    'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
    'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83,
    'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89,
    'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
    'Es': 99, 'Fm': 100, "None": 0,
}
atomic_number_to_element = {v: k for k, v in element_to_atomic_number.items()}
CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

def create_structure(crys_array_dict):
    # Extract data from input dictionary
    frac_coords = crys_array_dict['frac_coords']          # fractional coordinates
    occupancy_ratios = crys_array_dict['frac_occupancy']  # fractional occupancy
    atom_types = crys_array_dict['atom_types']            # major atom types (atomic numbers)
    lengths = crys_array_dict['lengths']                  # lattice lengths
    angles = crys_array_dict['angles']                    # lattice angles
    low_atomic_numbers = crys_array_dict['low_atomic_numbers']  # minor atom types

    # Create lattice
    lattice = Lattice.from_parameters(*lengths, *angles)

    # Convert atomic numbers to element symbols
    A_species = [atomic_number_to_element[number] for number in atom_types]
    B_species = [atomic_number_to_element[number] for number in low_atomic_numbers]

    # Initialize structure with major species and fractional coordinates
    structure = Structure(lattice, A_species, frac_coords)

    # Assign occupancy and mixed sites
    index = 0
    for (A, B) in zip(A_species, B_species):

        # Case 1: full occupancy by A
        if occupancy_ratios[index] >= 1.0:
            site_dict = structure[index].as_dict()
            site_dict['species'] = [{'element': A, 'oxidation_state': 0.0, 'occu': 1.0}]
            structure[index] = pymatgen.core.PeriodicSite.from_dict(site_dict)

        else:
            # Case 2: only one element (no mixing)
            if A == B or B == "None":
                site_dict = structure[index].as_dict()
                site_dict['species'] = [{'element': A, 'oxidation_state': 0.0,
                                         'occu': occupancy_ratios[index]}]
                structure[index] = pymatgen.core.PeriodicSite.from_dict(site_dict)

            # Case 3: A/B mixed site
            else:
                site_dict = structure[index].as_dict()
                site_dict['species'] = []
                site_dict['species'].append({'element': A, 'oxidation_state': 0.0,
                                             'occu': occupancy_ratios[index]})
                site_dict['species'].append({'element': B, 'oxidation_state': 0.0,
                                             'occu': int((1.0 - occupancy_ratios[index]) * 1e4) / 1e4})
                structure[index] = pymatgen.core.PeriodicSite.from_dict(site_dict)

        index += 1

    return structure


class Crystal(object):
    """Crystal object representing a single crystal instance.
    Stores coordinates, atom types, lattice, and computes:
    - structure
    - chemical composition
    - validity checks
    - structure fingerprints
    """

    def __init__(self, crys_array_dict):
        # Store raw data
        self.frac_coords = crys_array_dict['frac_coords']
        self.occupancy_ratios = crys_array_dict['frac_occupancy']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.low_atomic_numbers = crys_array_dict['low_atomic_numbers']
        self.dict = crys_array_dict

        # Convert one-hot atom types to integer atomic numbers (if needed)
        if len(self.atom_types.shape) > 1:
            self.dict['atom_types'] = np.argmax(self.atom_types, axis=-1) + 1
            self.atom_types = np.argmax(self.atom_types, axis=-1) + 1

        # Generate properties
        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()

    def get_structure(self):
        """Construct crystal structure and perform basic validity checks."""
        # Check lattice validity
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        elif np.isnan(self.lengths).any() or np.isnan(self.angles).any() or np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value'
        else:
            try:
                self.structure = create_structure(self.dict)
                self.constructed = True
                # Fully occupied version of the structure
                self.full_occupy_structure = build_structure(self.structure)
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'

            # Check volume
            if self.constructed and self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        """Compute reduced chemical composition."""
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        """Check composition and structural validity."""
        self.comp_valid = smact_validity(self.elems, self.comps)
        self.struct_valid = structure_validity(self.full_occupy_structure) if self.constructed else False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        """Compute composition and structural fingerprints."""
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)

        try:
            site_fps = [CrystalNNFP.featurize(self.full_occupy_structure, i)
                        for i in range(len(self.full_occupy_structure))]
        except Exception:
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return

        self.struct_fp = np.array(site_fps).mean(axis=0)


class RecEval(object):
    """Evaluate reconstruction quality of predicted crystals.
    Computes match rate and RMS distance between predicted and ground truth structures.
    """

    def __init__(self, pred_crys, gt_crys, gt_ctys_name, xrd_path,
                 stol=0.5, angle_tol=10, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.filename = gt_ctys_name
        self.xrd_path = xrd_path

    def get_match_rate_and_rms(self):

        def process_one(pred, gt, is_valid):
            """Compute RMS distance between predicted and ground truth structures."""
            gt = build_structure(gt)
            pred = build_structure(pred.structure)
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(pred, gt)
                return rms_dist[0] if rms_dist is not None else None
            except Exception:
                return None

        validity = [c.valid for c in self.preds]
        rms_dists = []
        similarity_list = []

        # Evaluate each predicted crystal
        for i in tqdm(range(len(self.preds))):
            rms = process_one(self.preds[i], self.gts[i], validity[i])
            if rms is None:
                continue

            # Compute XRD similarity
            xrd2 = get_xrd(self.preds[i].structure, self.filename[i])
            xrd1 = torch.load(join(self.xrd_path, self.filename[i].replace("cif", "pt")))
            similarity = cosine_similarity(np.array(xrd1), np.array(xrd2))

            # Only count samples with similarity > 0.7
            if similarity > 0.7:
                similarity_list.append(similarity)
                rms_dists.append(rms)

        # Save results
        result_path = "./MCSRNet/XTCSG/result"
        os.makedirs(result_path, exist_ok=True)

        rms_dists = np.array(rms_dists)
        similarity_list = np.array(similarity_list)

        match_rate = len(rms_dists) / len(self.preds)
        mean_rms_dist = rms_dists.mean() if rms_dists.size > 0 else float('nan')
        mean_similarity_list = similarity_list.mean() if similarity_list.size > 0 else "None"

        np.savetxt(
            join(result_path, "rms_sim.txt"),
            np.column_stack((similarity_list, rms_dists)),
            delimiter=",",
            header="similarity,rms",
            fmt="%.4f",
            comments=''
        )

        return {
            'match_rate': match_rate,
            'rms_dist': mean_rms_dist,
            'similarity_list': mean_similarity_list,
        }

    def get_metrics(self):
        return self.get_match_rate_and_rms()


class RecEvalBatch(object):
    """Batch evaluator for multiple sets of predicted crystals."""

    def __init__(self, pred_crys, gt_crys, gt_ctys_name, xrd_path,
                 stol=0.5, angle_tol=10, ltol=0.3):
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.filename = gt_ctys_name
        self.batch_size = len(self.preds)
        self.xrd_path = xrd_path

    def get_match_rate_and_rms(self):

        def process_one(pred, gt, is_valid):
            """Compute RMS distance for one predicted sample."""
            gt = build_structure(gt)
            pred = build_structure(pred.structure)
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(pred, gt)
                return rms_dist[0] if rms_dist is not None else None
            except Exception:
                return None

        rms_dists = []
        similarity_list = []

        # Loop over samples in batch
        for i in range(len(self.preds[0])):
            tmp_rms_dists = []
            tmp_similarity = []

            for j in range(self.batch_size):
                rmsd = process_one(self.preds[j][i], self.gts[i], self.preds[j][i].valid)
                if rmsd is None:
                    continue

                xrd2 = get_xrd(self.preds[j][i].structure, self.filename[i])
                xrd1 = torch.load(join(self.xrd_path, self.filename[i].replace("cif", "pt")))
                similarity = cosine_similarity(np.array(xrd1), np.array(xrd2))

                # Keep only similarity > 0.7
                if similarity > 0.7:
                    tmp_similarity.append(similarity)
                    tmp_rms_dists.append(rmsd)

            # Select best match in each sample group
            if len(tmp_similarity) != 0:
                max_value = max(tmp_similarity)
                similarity_list.append(max_value)

                indices = [k for k, val in enumerate(tmp_similarity) if val == max_value]
                min_rms_value = min(tmp_rms_dists[k] for k in indices)
                rms_dists.append(min_rms_value)

        # Save results
        result_path = "./MCSRNet/XTCSG/result"
        os.makedirs(result_path, exist_ok=True)

        rms_dists = np.array(rms_dists)
        similarity_list = np.array(similarity_list)

        match_rate = len(rms_dists) / len(self.preds[0])
        mean_rms_dist = rms_dists.mean() if rms_dists.size > 0 else float('nan')
        mean_similarity_list = similarity_list.mean() if similarity_list.size > 0 else "None"

        np.savetxt(
            join(result_path, f"rms_sim_{self.batch_size}.txt"),
            np.column_stack((similarity_list, rms_dists)),
            delimiter=",",
            header="similarity,rms",
            fmt="%.4f",
            comments=''
        )

        return {
            'match_rate': match_rate,
            'rms_dist': mean_rms_dist,
            'similarity_list': mean_similarity_list
        }

    def get_metrics(self):
        return self.get_match_rate_and_rms()



def get_file_paths(root_path, task, label='', suffix='pt'):
    if args.label == '':
        out_name = f'eval_{task}.{suffix}'
    else:
        out_name = f'eval_{task}_{label}.{suffix}'
    out_name = os.path.join(root_path, out_name)
    return out_name



def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    if batch_idx == -1:
        batch_size = data['frac_coords'].shape[0]
        crys_array_list = []
        for i in range(batch_size):
            tmp_crys_array_list = get_crystals_list(
                data['frac_coords'][i],
                data['atom_types'][i],
                data['frac_occupancy'][i],
                data['lengths'][i],
                data['angles'][i],
                data['num_atoms'][i],
                data['low_atomic_numbers'][i])
            crys_array_list.append(tmp_crys_array_list)
    elif batch_idx == -2:
        crys_array_list = get_crystals_list(
            data['frac_coords'],
            data['atom_types'],
            data['frac_occupancy'],
            data['lengths'],
            data['angles'],
            data['num_atoms'],
            data['low_atomic_numbers'])        
    else:
        crys_array_list = get_crystals_list(
            data['frac_coords'][batch_idx],
            data['atom_types'][batch_idx],
            data['frac_occupancy'][batch_idx],
            data['lengths'][batch_idx],
            data['angles'][batch_idx],
            data['num_atoms'][batch_idx],
            data['low_atomic_numbers'][batch_idx])

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['frac_occupancy'], batch['lengths'],
                batch['angles'], batch['num_atoms']),batch['low_atomic_numbers']
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.frac_occupancy, batch.lengths,
                batch.angles, batch.num_atoms, batch.low_atomic_numbers)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


def get_xrd(new_structure, name):
    the_stru = new_structure
    calculator = xrd.XRDCalculator()
    pattern = calculator.get_pattern(the_stru, two_theta_range=(5.0, 110))

    hkls, intensities = [v[0]['hkl'] for v in pattern.hkls], pattern.y
    scaled_intensities = []

    # Four Miller indicies in hexagonal systems
    if the_stru.lattice.is_hexagonal() == True:
        check = 0.0
        while check == 0.0:
            preferred_direction = [random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1]),
                                   random.choice([0, 1])]
            check = np.dot(np.array(preferred_direction), np.array(preferred_direction))  # Ensure 0-vector is not used

    # Three indicies are used otherwise
    else:
        check = 0.0
        while check == 0.0:
            preferred_direction = [random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1])]
            check = np.dot(np.array(preferred_direction),
                           np.array(preferred_direction))  # Make sure we don't have 0-vector

    for (hkl, peak) in zip(hkls, intensities):
        norm_1 = math.sqrt(np.dot(np.array(hkl), np.array(hkl)))
        norm_2 = math.sqrt(np.dot(np.array(preferred_direction), np.array(preferred_direction)))
        total_norm = norm_1 * norm_2
        texture_factor = abs(np.dot(np.array(hkl), np.array(preferred_direction)) / total_norm)
        max_texture = 0.5
        bound = 1.0 - max_texture
        texture_factor = bound + ( ( (1.0 - bound) / (1.0 - 0.0) ) * (texture_factor - 0.0) )
        scaled_intensities.append(peak * texture_factor)

    angles, intensities = pattern.x, scaled_intensities
    steps = np.linspace(5.0, 110, 5250)
    signals = np.zeros(steps.shape[0])
    possible_domains = np.linspace(1, 100)
    tau = random.choice(possible_domains)

    for i, ang in enumerate(angles):
        idx = np.argmin(np.abs(ang - steps))
        signals[idx] = intensities[i]

    conv = []
    for (ang, int) in zip(steps, signals):
        if int != 0:
            K = 0.9
            wavelength = 0.15406
            theta = math.radians(ang / 2.)
            beta = (K / wavelength) * (math.cos(theta) / tau)

            std_dev = beta / 2.35482

            gauss = [int * np.exp((-(val - ang) ** 2) / std_dev) for val in steps]
            conv.append(gauss)

    mixed_data = zip(*conv)
    all_I = []
    for values in mixed_data:
        noise = random.choice(np.linspace(-0.75, 0.75, 1000))
        all_I.append(sum(values) + noise)

    shifted_vals = np.array(all_I) - min(all_I)
    scaled_vals = 100 * np.array(shifted_vals) / max(shifted_vals)
    all_I = [val for val in scaled_vals]
    return all_I

def build_structure(crystal):
    elements = []
    for site in crystal:
        # Get species info: list like [(element, occupancy)]
        species_and_occu = list(site.species.items())

        if len(species_and_occu) == 1:
            # Single-element site
            element, occupancy = species_and_occu[0]
            elements.append(element.symbol)

        elif len(species_and_occu) >= 2:
            # Multi-element site: choose the one with highest occupancy
            species_and_occu_sorted = sorted(species_and_occu, key=lambda x: x[1], reverse=True)
            element1, occ1 = species_and_occu_sorted[0]
            elements.append(element1.symbol)

    # Build a new Structure using the major elements only
    structure = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=elements,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    return structure


def get_gt_crys_ori(csv_list):
    cif = csv_list[0] 
    filename = csv_list[1]
    structure = Structure.from_str(cif,fmt='cif')
    return structure, filename



def main(args):
    start_time = time.time()
    all_metrics = {}

    xrd_path = "./DCSP/clip/data//cif_oc/Perov_CaTiO3_0.7"+"/test/xrd"

    recon_file_path = get_file_paths(args.root_path, 'diff', args.label)
    batch_idx = -1 if args.multi_eval else 0
    crys_array_list, true_crystal_array_list = get_crystal_array_list(
        recon_file_path, batch_idx = batch_idx)
    if args.gt_file != '':
        csv = pd.read_csv(args.gt_file)
        csv_list = list(zip(csv['cif'], csv['filename']))
        gt_crys, gt_ctys_name = zip(*p_map(get_gt_crys_ori, csv_list))
    else:
        gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list)

    if not args.multi_eval:
        pred_crys = p_map(lambda x: Crystal(x), crys_array_list)
    else:
        pred_crys = []
        for i in range(len(crys_array_list)):
            print(f"Processing batch {i}")
            pred_crys.append(p_map(lambda x: Crystal(x), crys_array_list[i]))   

    if args.multi_eval:
        rec_evaluator = RecEvalBatch(pred_crys, gt_crys, gt_ctys_name, xrd_path)
    else:
        rec_evaluator = RecEval(pred_crys, gt_crys, gt_ctys_name, xrd_path)

    recon_metrics = rec_evaluator.get_metrics()

    all_metrics.update(recon_metrics)

    print(all_metrics)

    if args.label == '':
        metrics_out_file = 'eval_metrics.json'
    else:
        metrics_out_file = f'eval_metrics_{args.label}.json'
    metrics_out_file = os.path.join(args.root_path, metrics_out_file)

    # only overwrite metrics computed in the new run.
    if Path(metrics_out_file).exists():
        with open(metrics_out_file, 'r') as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(all_metrics)
            else:
                with open(metrics_out_file, 'w') as f:
                    json.dump(all_metrics, f)
        if isinstance(written_metrics, dict):
            with open(metrics_out_file, 'w') as f:
                json.dump(written_metrics, f)
    else:
        with open(metrics_out_file, 'w') as f:
            json.dump(all_metrics, f)
    print((time.time() - start_time)/3600)  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--gt_file',default='')
    parser.add_argument('--multi_eval',action='store_true')
    args = parser.parse_args()
    main(args)
