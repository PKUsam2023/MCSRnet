import time
import argparse
import torch
from pathlib import Path
from torch_geometric.data import Batch
from eval_utils import load_model, lattices_to_params_shape, recommand_step_lr


def diffusion(loader, model, num_evals, step_lr = 1e-5):

    frac_coords = []
    frac_occupancy = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    low_atomic_numbers = []
    for idx, batch in enumerate(loader):
        batch_start_time = time.time()  
        if torch.cuda.is_available():
            batch = batch.to('cuda')
        batch_frac_coords, batch_frac_occupancy, batch_num_atoms, batch_atom_types = [], [], [], []
        batch_lattices = []
        batch_low_atomic_numbers = []
        for eval_idx in range(num_evals):
            sample_start_time = time.time()  
            print(f'batch {idx} / {len(loader)}, sample {eval_idx} / {num_evals}')
            outputs, traj = model.sample(batch, step_lr = step_lr)#pl_modules.diffusion
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_frac_occupancy.append(outputs['frac_occupancy'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lattices.append(outputs['lattices'].detach().cpu())
            sample_time = (time.time() - sample_start_time) / 60
            batch_low_atomic_numbers.append(outputs['low_atomic_numbers'].detach().cpu())
            print(f'Sample {eval_idx} time: {sample_time:.2f} minutes')  
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        frac_occupancy.append(torch.stack(batch_frac_occupancy, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lattices.append(torch.stack(batch_lattices, dim=0))
        low_atomic_numbers.append(torch.stack(batch_low_atomic_numbers, dim=0))
        input_data_list = input_data_list + batch.to_data_list()
        batch_time = (time.time() - batch_start_time) / 60  
        print(f'Batch {idx} time: {batch_time:.2f} minutes')  

    frac_coords = torch.cat(frac_coords, dim=1)
    frac_occupancy = torch.cat(frac_occupancy, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lattices = torch.cat(lattices, dim=1)
    lengths, angles = lattices_to_params_shape(lattices)
    low_atomic_numbers = torch.cat(low_atomic_numbers, dim=1)
    input_data_batch = Batch.from_data_list(input_data_list)



    return (
        frac_coords, frac_occupancy, atom_types, lattices, lengths, angles, num_atoms, input_data_batch, low_atomic_numbers
    )



def main(args):
    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(model_path, load_data=True)

    if torch.cuda.is_available():
        device_ids = list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.cuda()

    print('Evaluate the diffusion model.')

    step_lr = args.step_lr if args.step_lr >= 0 else recommand_step_lr['csp' if args.num_evals == 1 else 'csp_multi'][args.dataset]

    start_time = time.time()
    (frac_coords, frac_occupancy, atom_types, lattices, lengths, angles, num_atoms, input_data_batch, low_atomic_numbers) = diffusion(
        test_loader, model, args.num_evals, step_lr)
        
    if args.label == '':
        diff_out_name = 'eval_diff.pt'
    else:
        diff_out_name = f'eval_diff_{args.label}.pt'

    torch.save({
        'eval_setting': args,
        'input_data_batch': input_data_batch,
        'frac_coords': frac_coords,
        'frac_occupancy': frac_occupancy,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lattices': lattices,
        'lengths': lengths,
        'angles': angles,
        'low_atomic_numbers': low_atomic_numbers,
        'time': time.time() - start_time,
    }, model_path / diff_out_name)  
    print((time.time() - start_time)/3600)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--step_lr', default=-1, type=float)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    main(args)
