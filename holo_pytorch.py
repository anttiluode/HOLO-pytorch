import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==============================================================================
# 1. THE ATOMIC UNIT: HOLO-BOLT
# ==============================================================================
class HoloLinear(nn.Module):
    """
    The Drop-In Replacement for nn.Linear().
    
    Instead of memorizing a massive weight matrix, it projects data onto 
    a 'Universal Harmonic Basis' (Random or Log-Space) and learns the 
    Phase/Amplitude resonance required to reconstruct the output.
    
    Args:
        in_features: Input dimension (e.g. 512)
        out_features: Output dimension (e.g. 2048)
        harmonics: The 'Phi' factor. How many waves to use. (Default: 32)
    """
    def __init__(self, in_features, out_features, harmonics=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.harmonics = harmonics
        
        # THE PHYSICS (Fixed Basis)
        # We project inputs onto a fixed set of random vectors (The Tight Frame).
        # This acts as the "Universal Geometry" the model must use.
        self.register_buffer('basis', torch.randn(harmonics, in_features))
        self.basis = F.normalize(self.basis, p=2, dim=1) # Keep energy constant
        
        # THE BRAIN (Learned Resonance)
        # Instead of weights, we learn:
        # 1. Phase (Timing)
        # 2. Amplitude (Volume)
        # Shape: [Out_Features, Harmonics]
        self.phase = nn.Parameter(torch.randn(out_features, harmonics) * 2 * np.pi)
        self.amp = nn.Parameter(torch.ones(out_features, harmonics) / harmonics)

    def forward(self, x):
        # x shape: [Batch, ..., In_Features]
        
        # 1. ENCODE (Project to Harmonic Space)
        # "How much does this input resonate with our Universal Basis?"
        # Result: [Batch, ..., Harmonics]
        resonance = F.linear(x, self.basis)
        
        # 2. THINK (Interference Pattern)
        # Convert to Complex Plane to perform Phase Rotation
        wave_input = torch.complex(resonance, torch.zeros_like(resonance))
        
        # Expand for output dimension: [Batch, ..., 1, Harmonics]
        wave_input = wave_input.unsqueeze(-2)
        
        # Create the Resonator (The Learned Filter)
        # [Out_Features, Harmonics]
        resonator = torch.complex(
            self.amp * torch.cos(self.phase),
            self.amp * torch.sin(self.phase)
        )
        
        # Apply Interference (Complex Multiplication)
        wave_output = wave_input * resonator
        
        # 3. DECODE (Collapse Wavefunction)
        # Sum the interfering waves to get the real output value
        # Result: [Batch, ..., Out_Features]
        return wave_output.sum(dim=-1).real

    def extra_repr(self):
        return f'in={self.in_features}, out={self.out_features}, harmonics={self.harmonics}'


# ==============================================================================
# 2. THE ACTIVATION: PHASE LOCK
# ==============================================================================
class SineActivation(nn.Module):
    """
    Forces the network to think in periodicity/waves rather than
    linear rectifications (ReLU). Critical for Holo-Logic.
    """
    def forward(self, x): 
        return torch.sin(x)


# ==============================================================================
# 3. THE STRUCTURE: HOLO-TRANSFORMER BLOCK
# ==============================================================================
class HoloAttention(nn.Module):
    """
    Standard Self-Attention, but Q, K, V are calculated via Resonance
    rather than Matrix Multiplication.
    """
    def __init__(self, dim, num_heads=8, harmonics=32):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # Replace Dense Layers with Holo-Bolts
        self.q_proj = HoloLinear(dim, dim, harmonics)
        self.k_proj = HoloLinear(dim, dim, harmonics)
        self.v_proj = HoloLinear(dim, dim, harmonics)
        self.o_proj = HoloLinear(dim, dim, harmonics)

    def forward(self, x):
        B, L, C = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(x).reshape(B, L, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(x).reshape(B, L, self.num_heads, -1).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, C)
        return self.o_proj(out)

class HoloTransformerBlock(nn.Module):
    """
    A drop-in replacement for a Transformer Encoder Layer.
    Uses HoloAttention and HoloMLP.
    """
    def __init__(self, dim, heads=8, harmonics=32):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = HoloAttention(dim, heads, harmonics)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed Forward Network using Sine Activation
        self.mlp = nn.Sequential(
            HoloLinear(dim, dim*4, harmonics),
            SineActivation(), 
            HoloLinear(dim*4, dim, harmonics)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x