# CSMF Project File Structure
## Version: CSMF-v1.0-Structure
## Based on: AMF-VI + Chapter 2 Work Packages

```
CSMF_project/
│
├── .gitignore                          # ✅ Keep from AMF-VI
├── README.md                           # 🔄 UPDATE for CSMF
├── requirements.txt                    # 🔄 UPDATE (add torchvision, POT)
│
├── configs/                            # 🆕 NEW - Level 0
│   ├── __init__.py
│   ├── mnist_config.py                 # 🆕 WP0.1 - All hyperparameters
│   ├── sr_config.py                    # 🆕 WP4.1 - Super-resolution config
│   └── sar_config.py                   # 🆕 WP4.2 - SAR despeckling config
│
├── csmf/                               # 🆕 NEW - Main CSMF package
│   ├── __init__.py
│   │
│   ├── conditioning/                   # 🆕 Level 1 - Conditioning components
│   │   ├── __init__.py
│   │   ├── film.py                     # 🆕 WP0.1 - FiLM layers: γ(h)⊙f + β(h)
│   │   └── conditioning_networks.py    # 🆕 WP0.1 - MNISTConditioner (4-6 layer CNN)
│   │
│   ├── flows/                          # 🔄 EXTEND from AMF-VI
│   │   ├── __init__.py
│   │   ├── base_flow.py                # ✅ Keep from AMF-VI
│   │   ├── realnvp.py                  # ✅ Keep (will extend to ConditionalRealNVP)
│   │   ├── maf.py                      # ✅ Keep (will extend to ConditionalMAF)
│   │   ├── rbig.py                     # ✅ Keep from AMF-VI
│   │   ├── conditional_realnvp.py      # 🆕 WP0.1 - Conditional version
│   │   ├── conditional_maf.py          # 🆕 WP0.1 - Conditional version
│   │   └── coupling_layers.py          # 🆕 Level 2 - ConditionalAffineCoupling
│   │
│   ├── physics/                        # 🆕 WP1 - Physics-based components
│   │   ├── __init__.py
│   │   ├── forward_models.py           # 🆕 WP1.1 - Blur, downsample, SAR ops
│   │   └── proximal.py                 # 🆕 WP1.1 - Proximal operators (closed-form)
│   │
│   ├── losses/                         # 🆕 WP2 - Hybrid objective
│   │   ├── __init__.py
│   │   ├── hybrid_loss.py              # 🆕 WP2.1 - NLL + consistency + transport
│   │   ├── calibration.py              # 🆕 WP2.4 - Energy score, CRPS
│   │   └── sliced_wasserstein.py       # 🆕 WP2.2 - Differentiable SW2
│   │
│   ├── models/                         # 🆕 Complete CSMF models
│   │   ├── __init__.py
│   │   ├── csmf.py                     # 🆕 Main CSMF class (mixture + conditioning)
│   │   └── amf_vi_baseline.py          # 🔄 Port from AMF-VI for comparison
│   │
│   └── utils/                          # 🆕 Utilities
│       ├── __init__.py
│       ├── metrics.py                  # 🆕 PSNR, SSIM, MMD, SW2, CRPS
│       └── visualization.py            # 🆕 Plotting for inverse problems
│
├── data/                               # 🔄 EXTEND from AMF-VI
│   ├── __init__.py
│   ├── data_generator.py               # ✅ Keep (2D synthetic datasets)
│   ├── visualize_data.py               # ✅ Keep from AMF-VI
│   ├── mnist_inverse.py                # 🆕 WP0.1 - MNIST inverse problems
│   ├── sr_dataset.py                   # 🆕 WP4.1 - Super-resolution (DIV2K/BSD)
│   └── sar_dataset.py                  # 🆕 WP4.2 - SAR despeckling
│
├── experiments/                        # 🆕 Experiment scripts
│   ├── __init__.py
│   ├── wp0_conditional_experts.py      # 🆕 WP0 - Train conditional flows
│   ├── wp1_consistency.py              # 🆕 WP1 - Test proximal steps
│   ├── wp2_hybrid_objective.py         # 🆕 WP2 - Train with hybrid loss
│   ├── wp3_ablations.py                # 🆕 WP3 - MNIST ablations
│   ├── wp4_imaging.py                  # 🆕 WP4 - SR and SAR experiments
│   └── train_csmf.py                   # 🆕 Main CSMF training script
│
├── tests/                              # 🔄 EXTEND from AMF-VI
│   ├── __init__.py
│   ├── test_flows.py                   # ✅ Keep from AMF-VI
│   ├── test_conditioning.py            # 🆕 WP0.4 - Unit tests for conditioning
│   ├── test_physics.py                 # 🆕 WP1 - Test forward models & proximal
│   ├── test_losses.py                  # 🆕 WP2 - Test hybrid loss components
│   └── test_integration.py             # 🆕 WP0.4 - End-to-end integration tests
│
├── scripts/                            # 🆕 Utility scripts
│   ├── download_datasets.py            # 🆕 Download DIV2K, BSD68
│   ├── preprocess_data.py              # 🆕 Create degraded images
│   └── evaluate_metrics.py             # 🆕 Compute all metrics on results
│
├── notebooks/                          # 🆕 Jupyter notebooks
│   ├── 01_explore_mnist_inverse.ipynb  # 🆕 Explore MNIST degradation
│   ├── 02_test_conditioning.ipynb      # 🆕 Test FiLM and conditioners
│   ├── 03_visualize_results.ipynb      # 🆕 Visualize reconstructions
│   └── 04_ablation_analysis.ipynb      # 🆕 Analyze ablation results
│
├── results/                            # 🔄 KEEP (auto-generated)
│   ├── models/                         # Saved model checkpoints
│   ├── plots/                          # Generated plots
│   └── metrics/                        # CSV files with metrics
│
├── docs/                               # 🆕 Documentation
│   ├── architecture.md                 # CSMF architecture overview
│   ├── wp0_guide.md                    # WP0 implementation guide
│   ├── wp1_guide.md                    # WP1 implementation guide
│   └── api_reference.md                # API documentation
│
└── legacy/                             # 🔄 Archive AMF-VI code
    └── amf_vi/                         # Move old AMF-VI here for reference
        ├── flows/
        ├── kde_kl_divergence.py
        ├── loss.py
        └── model.py
```

---

## 📋 File Status Legend

- ✅ **Keep from AMF-VI**: Use existing implementation as-is
- 🔄 **Extend/Update**: Modify existing file for CSMF
- 🆕 **New**: Create new file for CSMF

---

## 🎯 Priority Implementation Order (Based on Dependency Hierarchy)

### Week 1-2: WP0 Foundation (Priority 1)
```
1. configs/mnist_config.py              [Level 0]
2. csmf/conditioning/film.py            [Level 1]
3. csmf/conditioning/conditioning_networks.py [Level 1]
4. data/mnist_inverse.py                [Level 4]
5. csmf/flows/coupling_layers.py        [Level 2]
6. csmf/flows/conditional_realnvp.py    [Level 3]
7. csmf/flows/conditional_maf.py        [Level 3]
8. tests/test_conditioning.py           [Level 5]
```

### Week 3-4: WP1 Physics (Priority 2)
```
9. csmf/physics/forward_models.py
10. csmf/physics/proximal.py
11. tests/test_physics.py
```

### Week 5-6: WP2 Hybrid Loss (Priority 3)
```
12. csmf/losses/hybrid_loss.py
13. csmf/losses/calibration.py
14. csmf/losses/sliced_wasserstein.py
15. tests/test_losses.py
```

### Week 7-10: WP3-4 Full Implementation
```
16. csmf/models/csmf.py
17. experiments/train_csmf.py
18. experiments/wp3_ablations.py
19. data/sr_dataset.py
20. data/sar_dataset.py
```

---

## 📦 Updated Dependencies (requirements.txt)

```txt
# Core dependencies (from AMF-VI)
numpy>=1.21.0
matplotlib>=3.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
pandas>=1.3.0

# PyTorch ecosystem (NEW for CSMF)
torch>=1.9.0
torchvision>=0.10.0

# Optimal transport
POT>=0.8.0

# Image processing (NEW for CSMF)
Pillow>=8.3.0
opencv-python>=4.5.0

# Metrics (NEW for CSMF)
scikit-image>=0.18.0  # For PSNR, SSIM

# Testing
pytest>=6.2.0
pytest-cov>=2.12.0

# Notebooks
jupyter>=1.0.0
ipykernel>=6.0.0

# Visualization
seaborn>=0.11.0
plotly>=5.0.0  # For interactive plots
```

---

## 🔑 Key Differences from AMF-VI

### 1. **New Directory: `configs/`**
- Centralizes all hyperparameters
- Separate configs for MNIST, SR, SAR
- **Rationale**: Makes experiments reproducible and easy to modify

### 2. **New Directory: `csmf/conditioning/`**
- Core innovation: conditioning on observations y
- FiLM layers and conditioning networks
- **Rationale**: Enable flows to adapt based on degraded input

### 3. **New Directory: `csmf/physics/`**
- Forward models (blur, downsample, SAR)
- Proximal operators for measurement consistency
- **Rationale**: Physics-informed constraints for inverse problems

### 4. **New Directory: `csmf/losses/`**
- Hybrid objective (NLL + consistency + transport + calibration)
- **Rationale**: Balance likelihood with physics constraints

### 5. **Renamed Package: `amf_vi/` → `csmf/`**
- Main package reflects new focus on conditional flows
- **Rationale**: Clear distinction from unconditional AMF-VI

### 6. **New Directory: `experiments/`**
- Replaces `main/` with structured experiments
- One script per work package
- **Rationale**: Better organization for systematic ablations

### 7. **New Directory: `notebooks/`**
- Interactive exploration and visualization
- **Rationale**: Rapid prototyping and analysis

### 8. **Archive: `legacy/amf_vi/`**
- Preserves original AMF-VI code
- **Rationale**: Reference for comparison experiments

---

## 📝 Critical Implementation Notes

### From W0_1_dependency_hierarchy.md:
1. **NO circular dependencies**: Flow chart shows clean Level 0→5 hierarchy
2. **Test each level**: Don't move to next level until current passes
3. **Parallel development**: Level 1 files (film.py, conditioning_networks.py) can be built simultaneously
4. **Critical bottleneck**: coupling_layers.py needed before conditional_realnvp.py

### From Ch2_Plan.pdf:
1. **Prototype on MNIST first**: Fast 28×28 iterations before real imaging
2. **Three-stage training**: 
   - Stage A: Train experts with weak consistency
   - Stage B: Train gate with full hybrid loss
   - Stage C: Light joint fine-tuning
3. **Success metrics**: Lower ||Ax-y|| at matched NLL

### Version Tracking:
- Use format: `WP#.#-Component-v#.#`
- Example: `W0.1-FiLM-v1.0`, `W1.1-Proximal-v2.0`
- Document in file header comments

---

## 🚀 Quick Start After Structure Creation

```bash
# 1. Create directory structure
mkdir -p CSMF_project/{configs,csmf/{conditioning,flows,physics,losses,models,utils},data,experiments,tests,scripts,notebooks,results/{models,plots,metrics},docs,legacy}

# 2. Create __init__.py files
find CSMF_project -type d -exec touch {}/__init__.py \;

# 3. Copy AMF-VI files to legacy
cp -r amf_vi CSMF_project/legacy/

# 4. Copy reusable components
cp amf_vi/flows/{base_flow.py,realnvp.py,maf.py,rbig.py} CSMF_project/csmf/flows/
cp data/data_generator.py CSMF_project/data/

# 5. Install dependencies
pip install -r CSMF_project/requirements.txt

# 6. Start with WP0.1
cd CSMF_project
# Begin implementation following priority order above
```

---

## 📊 File Count Summary

| Category | AMF-VI | CSMF | New |
|----------|--------|------|-----|
| Core Package | 5 | 15 | +10 |
| Data | 2 | 5 | +3 |
| Configs | 0 | 3 | +3 |
| Experiments | 5 | 7 | +2 |
| Tests | 2 | 6 | +4 |
| Scripts | 0 | 3 | +3 |
| Notebooks | 0 | 4 | +4 |
| **Total** | **14** | **43** | **+29** |

---

## ✅ Next Steps

1. **Create this structure** using the bash commands above
2. **Start with configs/mnist_config.py** (Level 0)
3. **Implement FiLM and conditioning networks** (Level 1)
4. **Follow dependency hierarchy** strictly
5. **Test at each level** before proceeding

This structure ensures:
- ✅ Clean dependency hierarchy (no circular imports)
- ✅ Modular design (easy to test components)
- ✅ Clear separation of concerns
- ✅ Backward compatibility (legacy AMF-VI preserved)
- ✅ Scalable for WP0-WP5 implementation

**Ready to start implementation with minimal plan for each file!** 🚀