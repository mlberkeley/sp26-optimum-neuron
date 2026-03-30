import torch
import torch.nn.functional as F
import torchvision.io as io
from vae2_2 import Wan2_2_VAE
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


cap = cv2.VideoCapture("walrus.mp4")
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    frames.append(gray)

cap.release()

video_np = np.array(frames)  # shape = (N, H, W)

N = video_np.shape[0]
indices = [int(i * video_np.shape[0] / 9) for i in range(9)]
frames_9 = video_np[indices]  # (9, H, W)


frames_t = torch.from_numpy(frames_9).float() / 255.0  # (9, H, W, 3)
frames_t = frames_t.permute(0, 3, 1, 2)  # (9, 3, H, W)

H, W = frames_t.shape[2], frames_t.shape[3]
H_new = (H // 100) * 16
W_new = (W // 100) * 16
frames_t = F.interpolate(frames_t, size=(H_new, W_new))  # (9, 3, H_new, W_new)

frames_t = frames_t.permute(1, 0, 2, 3)  # (3, 9, H_new, W_new)
x = frames_t * 2 - 1  # [-1, 1]


vae = Wan2_2_VAE(
    vae_pth="./Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
    device="cpu",
    dtype=torch.float32,
)
# x = torch.randn(3, 9, 256, 256)
x = x.clone().requires_grad_(True)

with torch.enable_grad():
    z = vae.model.encode(x.unsqueeze(0), vae.scale).float().squeeze(0)
    x_hat = torch.clamp(
        vae.model.decode(z.unsqueeze(0), vae.scale).float().squeeze(0),
        -1.0, 1.0,
    )
    loss = F.mse_loss(x_hat, x)

loss.backward()

# checks
assert x.grad is not None, "gradients did not flow back to input"
assert torch.isfinite(x.grad).all(), "gradients contain NaN or Inf"
assert torch.isfinite(x_hat).all(), "reconstruction contains NaN or Inf"
assert not (x_hat == 0).all(), "reconstruction is all zeros"

print("loss:", loss.item())
print("grad mean:", x.grad.abs().mean().item())
print("all checks passed!")


os.makedirs("reconstruction_output", exist_ok=True)

# convert from [-1, 1] back to [0, 1]
orig = (x.detach() + 1) / 2        # (3, 9, H, W)
recon = (x_hat.detach() + 1) / 2   # (3, 9, H, W)

for i in range(9):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    orig_frame = orig[:, i, :, :].permute(1, 2, 0).numpy()   # (H, W, 3)
    recon_frame = recon[:, i, :, :].permute(1, 2, 0).numpy() # (H, W, 3)
    
    axes[0].imshow(orig_frame.clip(0, 1))
    axes[0].set_title(f"Original frame {i}")
    axes[1].imshow(recon_frame.clip(0, 1))
    axes[1].set_title(f"Reconstruction frame {i}")
    
    plt.savefig(f"reconstruction_output/frame_{i}.png")
    plt.close()

print("saved frames to reconstruction_output/")