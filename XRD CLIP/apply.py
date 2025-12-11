from tqdm.autonotebook import tqdm
import torch
import torch.nn.functional as F
from os.path import join
import multiprocessing
from model.config import CFG
from CXSP.model.cxsp_model import CXSPModel
from train import build_loaders


def get_xt_data(path_dir, filename):
    """Load and normalize XRD data for a CIF file."""
    xrd = torch.load(join(path_dir, "xrd", filename.replace("cif", "pt")))
    xrd = F.normalize(xrd, p=2, dim=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return xrd.to(device), filename


def get_cif_embeddings(model_path, mode):
    """Compute CIF embeddings for the dataset."""
    loader = build_loaders(mode=mode)
    model = CXSPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()

    embeddings, tokenizers = [], []

    with torch.no_grad():
        for batch in tqdm(loader):
            # Encode CIF features
            lattice = model.lattice_encoder(batch[0].float())
            atom = model.atom_encoder(batch[1].float())
            occupy = model.occupy_encoder(batch[3].float())
            cif_feat = torch.cat((lattice, atom, occupy), dim=1)
            embeddings.append(model.cif_projection(cif_feat))
            tokenizers += list(batch[4])

    return model, torch.cat(embeddings), tokenizers


def find_matches(model, cif_embeddings, xrd, tokenizer, cif_filenames, n):
    """Check if XRD matches top-N CIF embeddings."""
    global flagn
    global correct
    flagn += 1

    with torch.no_grad():
        xt = model.xrd_encoder(xrd.unsqueeze(0).float())
        xt_emb = model.xt_projection(xt)

    # Cosine similarity
    cif_norm = F.normalize(cif_embeddings, p=2, dim=-1)
    xt_norm = F.normalize(xt_emb, p=2, dim=-1)
    sim = xt_norm @ cif_norm.T
    top_idx = torch.topk(sim.squeeze(0), n).indices
    matches = [cif_filenames[i] for i in top_idx]

    return 1 if tokenizer in matches else 0


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    path_dir = "./MCSRNet/data/Perov_CaTiO3/test"
    model, cif_embeddings, cif_filenames = get_cif_embeddings("best.pt", "test")

    correct, flagn = 0, 0
    for fn in cif_filenames:
        xrd, tokenizer = get_xt_data(path_dir, fn)
        correct += find_matches(model, cif_embeddings, xrd, tokenizer, cif_filenames, n=1)

    accuracy = correct / len(cif_filenames)
    with open("./MCSRNet/CXSP/accuracy", "a+") as f:
        f.write(f"accuracy = {accuracy}\n")
