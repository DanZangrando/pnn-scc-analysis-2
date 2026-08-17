import os
import sys
import torch
from cellpose import models as cellpose_models

pnn_pkg_path = os.path.abspath("src/counting_perineuronal_nets")
if pnn_pkg_path not in sys.path:
    sys.path.insert(0, pnn_pkg_path)

def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cargando modelos de IA en dispositivo: {device}")

    # Cellpose PV Model
    model_pv = cellpose_models.CellposeModel(gpu=torch.cuda.is_available())

    # PNNloc (Faster R-CNN)
    from models.FasterRCNN import FasterRCNNWrapper
    model_loc = FasterRCNNWrapper(in_channels=1, out_channels=1, model_pretrained=False)
    ckpt_loc = "data/models/pnn_v2_fasterrcnn_640/best.pth"
    if os.path.exists(ckpt_loc):
        checkpoint_loc = torch.load(ckpt_loc, map_location=device)
        model_loc.load_state_dict(checkpoint_loc['model'])
    model_loc.to(device).eval()

    # PNNscore (ConvNet)
    from models.ConvNet import ConvNet
    model_score = ConvNet(in_channels=1, num_classes=1)
    ckpt_score = "data/models/pnn_v2_scoring_rank_learning/best.pth"
    if os.path.exists(ckpt_score):
        checkpoint_score = torch.load(ckpt_score, map_location=device)
        model_score.load_state_dict(checkpoint_score['model'])
    model_score.to(device).eval()

    return model_pv, model_loc, model_score, device
