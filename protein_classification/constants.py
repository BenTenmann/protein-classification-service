import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TYPES = {
    'LONG': torch.long,
    'FLOAT': torch.float32
}
