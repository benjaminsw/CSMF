# CSMF Implementation Quick Start Guide
## Version: CSMF-v1.0-QuickStart

This guide gets you from AMF-VI to CSMF in **minimal steps**.

---

## 🚀 Step-by-Step: Creating CSMF Structure

### Option 1: Automated (Recommended)

```bash
# 1. Run the structure generator
python create_csmf_structure.py

# 2. Navigate to project
cd CSMF_project

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify structure
tree -L 2
```

### Option 2: Manual

```bash
# Create directories
mkdir -p CSMF_project/{configs,csmf/{conditioning,flows,physics,losses,models,utils},data,experiments,tests,scripts,notebooks,results/{models,plots,metrics},docs,legacy}

# Create __init__.py files
find CSMF_project -type d -not -path "*/.*" -exec touch {}/__init__.py \;

# Copy AMF-VI files
cp -r amf-vi/amf_vi CSMF_project/legacy/
cp -r amf-vi/data/*.py CSMF_project/data/
```

---

## 📝 Week 1-2 Implementation Checklist (WP0.1)

### Day 1: Configuration & Foundation
- [ ] `configs/mnist_config.py` - Copy template from structure generator
- [ ] Define all hyperparameters (dataset, forward model, conditioning, training)
- [ ] **Test:** Import config and print values

### Day 2: FiLM Layer
- [ ] `csmf/conditioning/film.py`
- [ ] Implement `FiLM.__init__(feature_dim, condition_dim, hidden_dim)`
- [ ] Implement `FiLM.forward(features, conditioning)`
- [ ] **Test:** Create dummy inputs, check output shape

```python
# Quick test
import torch
from csmf.conditioning.film import FiLM

film = FiLM(feature_dim=128, condition_dim=64, hidden_dim=64)
features = torch.randn(32, 128)  # batch_size=32
conditioning = torch.randn(32, 64)
output = film(features, conditioning)
assert output.shape == (32, 128)
print("✅ FiLM works!")
```

### Day 3: Conditioning Network
- [ ] `csmf/conditioning/conditioning_networks.py`
- [ ] Implement `MNISTConditioner.__init__(input_channels, output_dim, n_layers)`
- [ ] Implement `MNISTConditioner.forward(y)` → h
- [ ] **Test:** Pass 28×28 image, get conditioning vector

```python
# Quick test
from csmf.conditioning.conditioning_networks import MNISTConditioner

conditioner = MNISTConditioner(input_channels=1, output_dim=64, n_layers=4)
y_degraded = torch.randn(8, 1, 28, 28)  # batch of degraded images
h = conditioner(y_degraded)
assert h.shape == (8, 64)
print("✅ MNISTConditioner works!")
```

### Day 4: MNIST Inverse Dataset
- [ ] `data/mnist_inverse.py`
- [ ] Implement `MNISTInverseDataset.__init__()` - loads MNIST
- [ ] Implement forward model: blur + downsample + noise
- [ ] Implement `__getitem__()` → (y_degraded, x_clean)
- [ ] **Test:** Load dataset, visualize pairs

```python
# Quick test
from data.mnist_inverse import MNISTInverseDataset
import matplotlib.pyplot as plt

dataset = MNISTInverseDataset(train=True, blur_sigma=1.0, noise_sigma=0.1)
y_deg, x_clean = dataset[0]

fig, axes = plt.subplots(1, 2)
axes[0].imshow(x_clean.squeeze(), cmap='gray')
axes[0].set_title('Clean')
axes[1].imshow(y_deg.squeeze(), cmap='gray')
axes[1].set_title('Degraded')
plt.show()
print("✅ MNIST Inverse Dataset works!")
```

### Day 5: Conditional Coupling Layer
- [ ] `csmf/flows/coupling_layers.py`
- [ ] Implement `ConditionalAffineCoupling` with FiLM
- [ ] Scale network: `s(x_A; h)` with FiLM in hidden layers
- [ ] Translation network: `t(x_A; h)` with FiLM
- [ ] **Test:** Forward + inverse, check invertibility

```python
# Quick test
from csmf.flows.coupling_layers import ConditionalAffineCoupling

coupling = ConditionalAffineCoupling(
    dim=28*28, 
    condition_dim=64,
    hidden_dim=128,
    mask=torch.tensor([1, 0] * (28*28//2))  # checkerboard
)

x = torch.randn(16, 28*28)
h = torch.randn(16, 64)

z, log_det = coupling.forward(x, h)
x_reconstructed = coupling.inverse(z, h)

error = (x - x_reconstructed).abs().max()
assert error < 1e-4, f"Invertibility error: {error}"
print("✅ ConditionalAffineCoupling works!")
```

### Week 2: Complete Conditional Flows

#### Day 6-7: ConditionalRealNVP
- [ ] `csmf/flows/conditional_realnvp.py`
- [ ] Stack `ConditionalAffineCoupling` layers
- [ ] Integrate `MNISTConditioner` to extract h from y
- [ ] Implement `forward_and_log_det(x, y)` and `inverse(z, y)`
- [ ] **Test:** Full forward-inverse cycle

```python
# Quick test
from csmf.flows.conditional_realnvp import ConditionalRealNVP

model = ConditionalRealNVP(
    data_dim=28*28,
    condition_dim=64,
    n_layers=4
)

x = torch.randn(8, 28*28)
y = torch.randn(8, 1, 28, 28)

z, log_det = model.forward_and_log_det(x, y)
x_recon = model.inverse(z, y)

error = (x - x_recon).abs().max()
print(f"Reconstruction error: {error}")
assert error < 1e-3
print("✅ ConditionalRealNVP works!")
```

#### Day 8-9: ConditionalMAF (Optional)
- [ ] `csmf/flows/conditional_maf.py`
- [ ] Adapt MAF to include conditioning h
- [ ] **Test:** Same as ConditionalRealNVP

#### Day 10: Unit Tests
- [ ] `tests/test_conditioning.py`
- [ ] Test FiLM layer
- [ ] Test MNISTConditioner
- [ ] Test ConditionalAffineCoupling
- [ ] Test ConditionalRealNVP
- [ ] Run: `pytest tests/test_conditioning.py -v`

---

## 🔬 Testing Strategy

### Unit Tests (tests/test_conditioning.py)
```python
import pytest
import torch
from csmf.conditioning.film import FiLM
from csmf.conditioning.conditioning_networks import MNISTConditioner
from csmf.flows.coupling_layers import ConditionalAffineCoupling

class TestFiLM:
    def test_forward_shape(self):
        film = FiLM(128, 64, 64)
        features = torch.randn(32, 128)
        conditioning = torch.randn(32, 64)
        output = film(features, conditioning)
        assert output.shape == (32, 128)
    
    def test_different_conditioning(self):
        """Different h should produce different outputs"""
        film = FiLM(128, 64, 64)
        features = torch.randn(32, 128)
        h1 = torch.randn(32, 64)
        h2 = torch.randn(32, 64)
        
        out1 = film(features, h1)
        out2 = film(features, h2)
        
        assert not torch.allclose(out1, out2)

class TestMNISTConditioner:
    def test_output_shape(self):
        conditioner = MNISTConditioner(1, 64, 4)
        y = torch.randn(8, 1, 28, 28)
        h = conditioner(y)
        assert h.shape == (8, 64)
    
    def test_different_images(self):
        """Different images should produce different h"""
        conditioner = MNISTConditioner(1, 64, 4)
        y1 = torch.randn(1, 1, 28, 28)
        y2 = torch.randn(1, 1, 28, 28)
        
        h1 = conditioner(y1)
        h2 = conditioner(y2)
        
        assert not torch.allclose(h1, h2)

class TestConditionalAffineCoupling:
    def test_invertibility(self):
        """x → z → x should recover x"""
        dim = 28*28
        coupling = ConditionalAffineCoupling(dim, 64, 128, 
                                            mask=torch.randint(0, 2, (dim,)))
        
        x = torch.randn(16, dim)
        h = torch.randn(16, 64)
        
        z, log_det = coupling.forward(x, h)
        x_recon = coupling.inverse(z, h)
        
        error = (x - x_recon).abs().max()
        assert error < 1e-4, f"Invertibility failed: {error}"
```

Run tests:
```bash
pytest tests/test_conditioning.py -v --tb=short
```

---

## 📊 Progress Tracking

Create a file `PROGRESS.md` to track implementation:

```markdown
# CSMF Implementation Progress

## WP0.1: Conditional Experts ✅ 🚧 ⏳

### Week 1 (Oct 1-7)
- [x] configs/mnist_config.py
- [x] csmf/conditioning/film.py
- [x] csmf/conditioning/conditioning_networks.py
- [x] data/mnist_inverse.py
- [ ] csmf/flows/coupling_layers.py
- [ ] csmf/flows/conditional_realnvp.py

### Week 2 (Oct 8-15)
- [ ] csmf/flows/conditional_maf.py
- [ ] tests/test_conditioning.py
- [ ] Integration test with full pipeline

### Blockers
- None currently

### Notes
- FiLM layer tested and working ✅
- MNISTConditioner produces reasonable h vectors ✅
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Dimension Mismatch in FiLM
**Error:** `RuntimeError: The size of tensor a (128) must match the size of tensor b (64)`

**Solution:** Check that `feature_dim` matches the actual feature dimension:
```python
# Wrong
film = FiLM(feature_dim=128, condition_dim=64)
features = torch.randn(32, 256)  # ❌ Wrong dimension

# Correct
film = FiLM(feature_dim=256, condition_dim=64)
features = torch.randn(32, 256)  # ✅ Correct
```

### Issue 2: Poor Invertibility in Coupling Layer
**Error:** Large reconstruction error (> 1e-2)

**Solution:** Check parameter order in constructor vs forward:
```python
# Common mistake
def forward(self, x, h):  # h is conditioning
    # But constructor expects (x, y)
    
# Fix: Be consistent with naming
def forward(self, x, conditioning):
```

### Issue 3: MNIST Inverse Dataset Returns Wrong Shapes
**Error:** Expected (1, 28, 28), got (28, 28)

**Solution:** Ensure proper unsqueezing:
```python
# Wrong
return x_clean, y_degraded  # Missing channel dim

# Correct
return x_clean.unsqueeze(0), y_degraded.unsqueeze(0)
```

---

## 📞 Quick Reference: Key Functions

### FiLM Layer
```python
film = FiLM(feature_dim, condition_dim, hidden_dim)
modulated = film(features, conditioning)
# Output shape: same as features
```

### MNISTConditioner
```python
conditioner = MNISTConditioner(input_channels=1, output_dim=64, n_layers=4)
h = conditioner(y_degraded)  # (B, 1, 28, 28) → (B, 64)
```

### ConditionalAffineCoupling
```python
coupling = ConditionalAffineCoupling(dim, condition_dim, hidden_dim, mask)
z, log_det = coupling.forward(x, h)
x_recon = coupling.inverse(z, h)
```

### MNISTInverseDataset
```python
dataset = MNISTInverseDataset(train=True, blur_sigma=1.0, noise_sigma=0.1)
y_degraded, x_clean = dataset[idx]
# Returns: (1, H, W), (1, H, W)
```

---

## ✅ Definition of Done (WP0.1)

- [ ] All Level 0-1 files created and tested
- [ ] Invertibility test passes: `max |x - f^{-1}(f(x))| < 1e-5`
- [ ] Conditioning determinism: different y → different h
- [ ] Unit tests pass: `pytest tests/test_conditioning.py`
- [ ] Can train a simple conditional flow on MNIST
- [ ] Documentation updated
- [ ] Code reviewed and merged

---

## 🎯 Next Steps After WP0.1

Once WP0.1 is complete:
1. Move to WP1: Physics-based corrections (Week 3-4)
2. Implement forward models and proximal operators
3. Test measurement consistency

See `CSMF_FILE_STRUCTURE.md` for full roadmap.

---

**Questions?** Refer to:
- Full structure: `CSMF_FILE_STRUCTURE.md`
- Dependencies: `W0_1_dependency_hierarchy.md`
- Work packages: `CH2_Workpackage_Implementation_Details.pdf`