import torch, logging
logging.basicConfig(level=logging.WARNING)
from csmf.flows.conditional_realnvp import ConditionalRealNVP
from scripts.preprocess_mnist import create_precomputed_dataloaders

train_dl, val_dl, test_dl = create_precomputed_dataloaders()
x, y = next(iter(val_dl))
x, y = x[:16], y[:16]

model = ConditionalRealNVP()
ckpt = torch.load('checkpoints/expert_0_ConditionalRealNVP.pth', map_location='cpu')
model.load_state_dict({
    k.replace('experts.0.', ''): v
    for k, v in ckpt['state_dict'].items()
    if k.startswith('experts.0.')
})
model.eval()

with torch.no_grad():
    z_final, z_list, log_det, log_prob = model(x, y)
    print(f'z_final mean={z_final.mean():.4e}, std={z_final.std():.4e}')
    