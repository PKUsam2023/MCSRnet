import torch
class CFG:
    debug = False

    train_path = "./MCSRNet/data/Perov_CaTiO3/train"
    test_path = "./MCSRNet/data/Perov_CaTiO3/test"
    val_path = "./MCSRNet/data/Perov_CaTiO3/val"

    batch_size = 128
    num_workers = 4
    head_lr = 1e-4  #3
    tem_encoder_lr = 1e-4
    cif_encoder_lr = 1e-4 #3
    xrd_encoder_lr = 1e-4 #3
    weight_decay = 1e-3  #3
    patience = 3  #1
    factor = 0.8  #0.8
    epochs = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    cif_embedding = 144
    xt_embedding = 1024

    trainable = True # for both image encoder and text encoder
    temperature = 1.0


    # for projection head; used for both image and text encoders
    projection_dim = 256 
    dropout = 0.2   #0.1 


 


