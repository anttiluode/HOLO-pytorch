import argparse
import os
import sys
import types
import threading
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageTk, ImageEnhance
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import time

# ===============================
# Environment Setup & Monkey-Patch for Triton
# ===============================
os.environ["DIFFUSERS_NO_IP_ADAPTER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    import triton.runtime
except ImportError:
    sys.modules["triton"] = types.ModuleType("triton")
    sys.modules["triton.runtime"] = types.ModuleType("triton.runtime")
    import triton.runtime

if not hasattr(triton.runtime, "Autotuner"):
    class DummyAutotuner:
        def __init__(self, *args, **kwargs):
            pass
        def tune(self, *args, **kwargs):
            return None
    triton.runtime.Autotuner = DummyAutotuner

# ===============================
# Imports from diffusers
# ===============================
from diffusers import StableVideoDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 1. THE HOLO-BOLT (Holographic Resonance Layers)
# ==============================================================================
class SineActivation(nn.Module):
    """
    Forces the network to think in periodicity/waves rather than
    linear rectifications (ReLU). Critical for Holo-Logic.
    """
    def forward(self, x): return torch.sin(x)

class HoloLinear(nn.Module):
    """
    The Harmonic Resonator.
    Replaces Matrix Multiplication with Geometric Phase Interference.
    """
    def __init__(self, in_features, out_features, harmonics=32):
        super().__init__()
        self.harmonics = harmonics
        
        # FIXED PHYSICS: Project onto Universal Basis (Random Tight Frame)
        self.register_buffer('basis', torch.randn(harmonics, in_features))
        self.basis = F.normalize(self.basis, p=2, dim=1)
        
        # LEARNED MIND: Tune Phase and Amplitude to resonate
        self.phase = nn.Parameter(torch.randn(out_features, harmonics) * 2 * np.pi)
        self.amp = nn.Parameter(torch.ones(out_features, harmonics) / harmonics)

    def forward(self, x):
        # 1. ENCODE: How much does input resemble the Universal Harmonics?
        resonance = F.linear(x, self.basis) 
        
        # 2. THINK: Apply Phase Interference in Complex Plane
        wave = torch.complex(resonance, torch.zeros_like(resonance)).unsqueeze(-2)
        resonator = torch.complex(self.amp * torch.cos(self.phase), self.amp * torch.sin(self.phase))
        
        # 3. DECODE: Collapse wave function to reality
        return (wave * resonator).sum(dim=-1).real

class HoloImageBlock(nn.Module):
    """
    Applies Holo-Logic to Images (pixel-wise).
    Treats every pixel as a 'token' in the Harmonic Universe.
    """
    def __init__(self, channels, harmonics=32):
        super().__init__()
        # We process the 'channels' dimension as the feature vector
        self.holo = HoloLinear(channels, channels, harmonics)
        self.act = SineActivation()
    
    def forward(self, x):
        # x: [Batch, Channels, Height, Width]
        # Permute to [Batch, Height, Width, Channels] for Linear layer
        h = x.permute(0, 2, 3, 1)
        h = self.holo(h)
        h = self.act(h)
        # Permute back
        return h.permute(0, 3, 1, 2)

# ===============================
# 2. ADAPTIVE VAE (Now Powered by Holo-Tech)
# ===============================
class AdaptiveEncoderConv(nn.Module):
    def __init__(self):
        super(AdaptiveEncoderConv, self).__init__()
        # Downsample: 512x512 -> 256 -> 128 -> 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)    # 512 -> 256
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)   # 256 -> 128
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 128 -> 64
        
        # --- HOLO-BOLT INSERTION ---
        # Before we compress to the final latent code, we run it through the Holo-Glass.
        # This organizes the features using 64 Harmonics.
        self.holo_neck = HoloImageBlock(256, harmonics=64)
        # ---------------------------

        self.conv4 = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)      # keep 64x64, output channels=4
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Apply Harmonic Reasoning
        x = self.holo_neck(x)
        
        latent = self.conv4(x)  # Expected shape: [batch, 4, 64, 64]
        return latent

class AdaptiveDecoderConv(nn.Module):
    def __init__(self):
        super(AdaptiveDecoderConv, self).__init__()
        # Upsample: 64x64 -> 128x128 -> 256x256 -> 512x512.
        self.conv_trans1 = nn.ConvTranspose2d(4, 256, kernel_size=3, stride=1, padding=1)   # 64 remains
        
        # --- HOLO-BOLT INSERTION ---
        # Immediately after unpacking the latent, we resonate.
        self.holo_neck = HoloImageBlock(256, harmonics=64)
        # ---------------------------

        self.conv_trans2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)   # 64 -> 128
        self.conv_trans3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)    # 128 -> 256
        self.conv_trans4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)      # 256 -> 512
        self.relu = nn.ReLU()
    
    def forward(self, latent):
        x = self.relu(self.conv_trans1(latent))
        
        # Apply Harmonic Reasoning
        x = self.holo_neck(x)
        
        x = self.relu(self.conv_trans2(x))
        x = self.relu(self.conv_trans3(x))
        recon = torch.sigmoid(self.conv_trans4(x))  # Output in [0,1]
        return recon

# ===============================
# 3. TRAINING ENGINE
# ===============================
class AdaptiveVAETrainer:
    def __init__(self, encoder, decoder, teacher_vae):
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_vae = teacher_vae  # teacher_vae from diffusers pipeline (pipe.vae)
        
        # Note: We are now optimizing Holo Parameters (Phase/Amp) alongside Standard Weights
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-4)
        
        # L1 Loss is often better for Holo-Phase locking than MSE, but MSE is standard for VAEs.
        # We stick to MSE for compatibility with the Teacher VAE.
        self.loss_fn = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_on_frame(self, image_tensor):
        # image_tensor: FP32 tensor [1, 3, 512, 512]
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()
        
        # Teacher VAE (Standard AI) -> Gives us the "Truth"
        with torch.no_grad():
            teacher_latent = self.teacher_vae.encode(image_tensor.half()).latent_dist.sample().float()
            decoded = self.teacher_vae.decode(teacher_latent.half(), num_frames=1).sample
            teacher_decoded = ((decoded / 2 + 0.5).clamp(0, 1)).float()
            
        # Student VAE (Holo AI) -> Tries to replicate it using Harmonics
        with torch.cuda.amp.autocast():
            pred_latent = self.encoder(image_tensor)
            latent_loss = self.loss_fn(pred_latent, teacher_latent)
            pred_image = self.decoder(pred_latent)
            image_loss = self.loss_fn(pred_image, teacher_decoded)
            loss = latent_loss + image_loss
            
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

# ===============================
# 4. GUI & APP LOGIC
# ===============================
class LatentVideoFilter:
    def __init__(self, master):
        self.master = master
        self.master.title("Live Holo-Webcam (Adaptive VAE)")
        self.device = device
        
        # Load the teacher VAE
        print("Loading Stable Video Diffusion (Teacher)...")
        self.video_pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16
        ).to(self.device)
        
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
        
        self.cap = None
        self.camera_index = 0
        
        self.setup_gui()
        
        # Init Holo VAE
        print("Initializing Holographic VAE...")
        self.adaptive_encoder = AdaptiveEncoderConv().to(self.device)
        self.adaptive_decoder = AdaptiveDecoderConv().to(self.device)
        
        self.adaptive_trainer = AdaptiveVAETrainer(self.adaptive_encoder, self.adaptive_decoder, self.video_pipe.vae)
        
        self.teach_mode = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
        self.training_thread.start()
        
        self.update_video()
    
    def setup_gui(self):
        control_frame = tk.Frame(self.master)
        control_frame.pack(side='top', fill='x', padx=10, pady=5)
        
        self.teach_button = tk.Button(control_frame, text="Start Teach Mode", command=self.toggle_teach_mode)
        self.teach_button.pack(side='left', padx=5)
        
        self.save_button = tk.Button(control_frame, text="Save Holo-Model", command=self.save_model)
        self.save_button.pack(side='left', padx=5)
        
        self.load_button = tk.Button(control_frame, text="Load Holo-Model", command=self.load_model)
        self.load_button.pack(side='left', padx=5)
        
        self.video_label = tk.Label(self.master)
        self.video_label.pack(padx=10, pady=10)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(self.master, textvariable=self.status_var, relief='sunken', anchor='w')
        self.status_label.pack(side='bottom', fill='x')
    
    def toggle_teach_mode(self):
        self.teach_mode = not self.teach_mode
        if self.teach_mode:
            self.teach_button.config(text="Stop Teach Mode")
            self.status_var.set("Teach mode active (Holo-Tuning...)")
        else:
            self.teach_button.config(text="Start Teach Mode")
            self.status_var.set("Teach mode paused")
    
    def save_model(self):
        filename = filedialog.asksaveasfilename(title="Save Holo VAE", defaultextension=".pth",
                                                  filetypes=[("Holo files", "*.pth")])
        if filename:
            torch.save({
                'encoder': self.adaptive_encoder.state_dict(),
                'decoder': self.adaptive_decoder.state_dict(),
            }, filename)
            self.status_var.set(f"Holo-Model saved to {filename}")
    
    def load_model(self):
        filename = filedialog.askopenfilename(title="Load Holo VAE",
                                              filetypes=[("Holo files", "*.pth")])
        if filename:
            checkpoint = torch.load(filename, map_location=self.device)
            self.adaptive_encoder.load_state_dict(checkpoint['encoder'])
            self.adaptive_decoder.load_state_dict(checkpoint['decoder'])
            self.status_var.set(f"Holo-Model loaded from {filename}")
    
    def training_loop(self):
        while True:
            if self.teach_mode and self.latest_frame is not None:
                with self.frame_lock:
                    frame = self.latest_frame.copy()
                try:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    transform = T.Compose([
                        T.Resize((512, 512)),
                        T.ToTensor(),
                    ])
                    image_tensor = transform(image).unsqueeze(0).to(self.device)
                    loss = self.adaptive_trainer.train_on_frame(image_tensor)
                    self.status_var.set(f"Teach mode active | Phase Loss: {loss:.4f}")
                except Exception as e:
                    self.status_var.set(f"Training error: {e}")
            time.sleep(0.05) # Faster training loop
    
    def update_video(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                try:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    transform = T.Compose([
                        T.Resize((512, 512)),
                        T.ToTensor(),
                    ])
                    image_tensor = transform(image).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        latent = self.adaptive_encoder(image_tensor)
                        recon = self.adaptive_decoder(latent)
                    recon_np = recon.cpu().squeeze(0).permute(1, 2, 0).numpy()
                    recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)
                    display_frame = cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    self.status_var.set(f"Processing error: {e}")
                    display_frame = frame
                image_pil = Image.fromarray(display_frame)
                photo = ImageTk.PhotoImage(image=image_pil)
                self.video_label.config(image=photo)
                self.video_label.image = photo
            self.master.after(30, self.update_video)
    
    def run(self):
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.master.mainloop()
    
    def on_closing(self):
        if self.cap is not None:
            self.cap.release()
        self.master.destroy()

def main():
    root = tk.Tk()
    app = LatentVideoFilter(root)
    app.run()

if __name__ == "__main__":
    main()