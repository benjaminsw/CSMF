# CSMF Project
**Conditional Sequential Mixture of Flows for Imaging Inverse Problems**

## Project Structure

```
CSMF_project/
├── configs/              # Configuration files for different tasks
│   ├── mnist_config.py   # WP0-WP3: MNIST experiments
│   ├── sr_config.py      # WP4.1: Super-resolution
│   └── sar_config.py     # WP4.2: SAR despeckling
│
├── csmf/                 # Main CSMF package
│   ├── conditioning/     # FiLM layers, conditioning networks (WP0.1)
│   ├── flows/           # Flow architectures (RealNVP, MAF, RBIG)
│   ├── physics/         # Measurement-consistency layers (WP1)
│   ├── losses/          # Hybrid objectives (WP2)
│   ├── models/          # Complete CSMF models
│   └── utils/           # Utilities and metrics
│
├── data/                # Dataset loaders and generators
├── experiments/         # Training scripts for each WP
├── tests/              # Unit tests
├── results/            # Saved models and plots
└── legacy/             # Original AMF-VI code for reference

```

## Work Packages Timeline

- **WP0** (Oct 1-15, 2025): Conditional experts, gating, baselines
- **WP1** (Oct 16 - Nov 15): Measurement-consistency layers
- **WP2** (Nov 16 - Dec 15): Hybrid objective & training schedule
- **WP3** (Dec 16 - Jan 20, 2026): MNIST ablations
- **WP4** (Jan 21 - Mar 10): Optical SR & SAR despeckling
- **WP5** (Mar 11-31): First outcomes & write-up

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start with WP0.1 - Implement conditioning:**
   - `csmf/conditioning/film.py` - FiLM layers
   - `csmf/conditioning/conditioning_networks.py` - MNISTConditioner

3. **Follow W0.1 dependency hierarchy:**
   - Level 0: configs/mnist_config.py ✓
   - Level 1: conditioning/ (FiLM, MNISTConditioner)
   - Level 2: flows/coupling_layers.py
   - Level 3: flows/conditional_flows.py
   - Level 4: data/datasets.py
   - Level 5: tests/

## Setup Info

- **Setup date:** 2025-12-08 16:26:24
- **Target directory:** /home/benjamin/Documents/CSMF
- **In-place mode:** Yes
- **AMF-VIJ source:** https://github.com/benjaminsw/AMF-VIJ

## Key References

- W0.1 Dependency Hierarchy: See project documentation
- Chapter 2 Implementation Details: See PDF guides
- Original AMF-VI paper: See legacy/amf_vi/
