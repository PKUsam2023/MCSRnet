from tqdm.autonotebook import tqdm
import torch
import torch.nn.functional as F
from os.path import join
import multiprocessing
import time
from model.config import CFG
from model.xtsp_model import XTSPModel
from train import build_loaders
import torchvision.transforms as transforms
import cv2 as cv


def get_xt_data(path_dir, filename):
    """Load XRD tensor and corresponding TEM images for a CIF file."""
    xrd_path = join(path_dir, "xrd")
    tem_path = join(path_dir, "tem", filename.replace(".cif", ""))
    transf = transforms.ToTensor()
    
    # Load and normalize XRD
    xrd = torch.load(join(xrd_path, filename.replace("cif", "pt")))
    xrd = F.normalize(xrd, p=2, dim=0)
    
    # Load 7 TEM images and concatenate
    tem_imgs = ['100.png', '010.png', '001.png', '110.png', '101.png', '011.png', '111.png']
    tem_list = []
    for img_name in tem_imgs:
        img = cv.imread(join(tem_path, img_name), cv.IMREAD_GRAYSCALE)
        tem_list.append(transf(img))
    tem = torch.cat(tem_list, dim=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return xrd.to(device), tem.to(device), filename


def get_cif_embeddings(model_path, mode):
    """Compute all CIF embeddings for the dataset."""
    loader = build_loaders(mode=mode)
    model = XTSPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()

    embeddings, tokenizers = [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            # Encode lattice, atom, and occupancy
            lattice = model.lattice_encoder(batch[0].float())
            atom = model.atom_encoder(batch[1].float())
            occupy = model.occupy_encoder(batch[3].float())
            cif_feat = torch.cat((lattice, atom, occupy), dim=1)
            embeddings.append(model.cif_projection(cif_feat))
            tokenizers += list(batch[5])

    return model, torch.cat(embeddings), tokenizers


def find_matches(model, cif_embeddings, xrd, tem, tokenizer, cif_filenames, mode, n):
    """Check if XRD+TEM matches top-N CIF embeddings."""
    global flagn, correct
    flagn += 1

    with torch.no_grad():
        # Encode TEM and XRD, combine and project
        tem_feat = model.tem_encoder(tem.unsqueeze(0).float())
        xrd_feat = model.xrd_encoder(xrd.unsqueeze(0).float())
        xt_emb = model.xt_projection(torch.cat([tem_feat, xrd_feat], dim=1))

    # Cosine similarity with CIF embeddings
    cif_norm = F.normalize(cif_embeddings, p=2, dim=-1)
    xt_norm = F.normalize(xt_emb, p=2, dim=-1)
    sim = xt_norm @ cif_norm.T
    top_idx = torch.topk(sim.squeeze(0), n).indices
    matches = [cif_filenames[i] for i in top_idx]

    return 1 if tokenizer in matches else 0


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    start = time.time()

    path_dir = "./MCSRNet/data/Perov_CaTiO3/test"
    model, cif_embeddings, cif_filenames = get_cif_embeddings("best.pt", "test")

    correct, flagn = 0, 0
    for fn in cif_filenames:
        xrd, tem, tokenizer = get_xt_data(path_dir, fn)
        correct += find_matches(model, cif_embeddings, xrd, tem, tokenizer, cif_filenames, "test", n=10)

    accuracy = correct / len(cif_filenames)
    with open("./MCSRNet/XTSP/accuracy", "a+") as f:
        f.write(f"accuracy = {accuracy}\n")
