import torch
import torch.nn.functional as F
import torchvision.io as io
from vae2_2 import Wan2_2_VAE
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


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
frames_t = torch.rot90(frames_t, k=1, dims=(2, 3))

H, W = frames_t.shape[2], frames_t.shape[3]
H_new = (H // 100) * 16
W_new = (W // 100) * 16
H_new, W_new = 256, 256

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



os.makedirs("reconstruction_output", exist_ok=True)

orig = x.detach().clamp(-1, 1).cpu()       # (3, 9, H, W)
recon = x_hat.detach().clamp(-1, 1).cpu()  # (3, 9, H, W)

def to_rgb_uint8(t: torch.Tensor) -> np.ndarray:
    # t: (3, H, W) in [-1, 1]
    return ((t + 1.0) * 0.5 * 255.0).byte().permute(1, 2, 0).numpy()

diff_gain = 4.0

for i in range(orig.shape[1]):
    a = to_rgb_uint8(orig[:, i])   # original
    b = to_rgb_uint8(recon[:, i])  # reconstruction

    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.float32)
    d_vis = np.clip(diff * diff_gain, 0, 255).astype(np.uint8)

    h, w, _ = a.shape
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(Image.fromarray(a), (0, 0))
    canvas.paste(Image.fromarray(b), (w, 0))
    canvas.paste(Image.fromarray(d_vis), (2 * w, 0))

    canvas.save(f"reconstruction_output/frame_{i}.png")

print("saved triptych frames to reconstruction_output/")