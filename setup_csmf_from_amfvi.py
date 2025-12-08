#!/usr/bin/env python3
"""
CSMF Project Setup from AMF-VIJ Repository
Version: CSMF-v1.1-Setup-InPlace
Author: Based on W0.1 dependency hierarchy

Clones AMF-VIJ repository and organizes files into CSMF structure
according to the project specifications.

USAGE:
  Run inside CSMF folder:
    python setup_csmf_from_amfvi.py
  
  Run with auto-confirm:
    python setup_csmf_from_amfvi.py --yes
  
  Run from outside (create new directory):
    python setup_csmf_from_amfvi.py --target /path/to/new/csmf
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class CSMFSetup:
    """Manages the CSMF project setup from AMF-VIJ repository"""
    
    def __init__(self, target_dir: str = "."):
        self.target_dir = Path(target_dir).resolve()
        self.in_place = (target_dir == "." or target_dir == self.target_dir.name)
        self.repo_url = "https://github.com/benjaminsw/AMF-VIJ"
        
        # Put temp clone in parent directory to avoid conflicts
        self.temp_clone_dir = self.target_dir.parent / "temp_amfvi_clone"
        
        # Track operations
        self.operations = {
            'success': [],
            'failed': [],
            'skipped': [],
            'warnings': []
        }
        
        # Define CSMF structure
        self.structure = {
            'configs': [],
            'csmf': {
                'conditioning': [],
                'flows': [],
                'physics': [],
                'losses': [],
                'models': [],
                'utils': []
            },
            'data': [],
            'experiments': [],
            'tests': [],
            'scripts': [],
            'notebooks': [],
            'results': {
                'models': [],
                'plots': [],
                'metrics': []
            },
            'docs': [],
            'legacy': {
                'amf_vi': []
            }
        }
        
    def log(self, message: str, level: str = 'info'):
        """Log messages with color coding"""
        colors = {
            'info': Colors.OKBLUE,
            'success': Colors.OKGREEN,
            'warning': Colors.WARNING,
            'error': Colors.FAIL,
            'header': Colors.HEADER
        }
        color = colors.get(level, '')
        print(f"{color}{message}{Colors.ENDC}")
        
    def create_directory_structure(self) -> bool:
        """Create the CSMF directory structure"""
        self.log("\n📁 Creating CSMF directory structure...", 'header')
        
        try:
            # Create main directory if needed
            if not self.in_place:
                self.target_dir.mkdir(exist_ok=True)
                self.operations['success'].append(f"Created {self.target_dir}")
            else:
                self.log(f"   Using existing directory: {self.target_dir}", 'info')
            
            # Create all subdirectories
            dirs_to_create = [
                'configs',
                'csmf/conditioning',
                'csmf/flows',
                'csmf/physics',
                'csmf/losses',
                'csmf/models',
                'csmf/utils',
                'data',
                'experiments',
                'tests',
                'scripts',
                'notebooks',
                'results/models',
                'results/plots',
                'results/metrics',
                'docs',
                'legacy/amf_vi'
            ]
            
            for dir_path in dirs_to_create:
                full_path = self.target_dir / dir_path
                full_path.mkdir(parents=True, exist_ok=True)
                
                # Create __init__.py for Python packages
                if any(x in dir_path for x in ['csmf', 'configs', 'data', 'experiments', 'tests', 'scripts']):
                    init_file = full_path / '__init__.py'
                    if not init_file.exists():
                        init_file.touch()
                        self.operations['success'].append(f"Created {dir_path}/__init__.py")
                
                self.operations['success'].append(f"Created directory {dir_path}")
            
            self.log(f"✅ Created {len(dirs_to_create)} directories", 'success')
            return True
            
        except Exception as e:
            self.log(f"❌ Failed to create directories: {e}", 'error')
            self.operations['failed'].append(f"Directory creation: {str(e)}")
            return False
    
    def clone_repository(self) -> bool:
        """Clone the AMF-VIJ repository"""
        self.log("\n📥 Cloning AMF-VIJ repository...", 'header')
        
        try:
            # Check if temp directory already exists
            if self.temp_clone_dir.exists():
                self.log(f"⚠️  Temp directory exists: {self.temp_clone_dir}", 'warning')
                self.log("   Removing old clone...", 'info')
                shutil.rmtree(self.temp_clone_dir)
            
            # Clone repository
            self.log(f"   Cloning from {self.repo_url}...", 'info')
            result = subprocess.run(
                ['git', 'clone', self.repo_url, str(self.temp_clone_dir)],
                capture_output=True,
                text=True,
                check=True
            )
            
            self.operations['success'].append(f"Cloned repository to {self.temp_clone_dir}")
            self.log(f"✅ Repository cloned successfully", 'success')
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Git clone failed: {e.stderr}", 'error')
            self.operations['failed'].append(f"Git clone: {e.stderr}")
            return False
        except FileNotFoundError:
            self.log(f"❌ Git not found. Please install git first.", 'error')
            self.operations['failed'].append("Git not installed")
            return False
        except Exception as e:
            self.log(f"❌ Unexpected error: {e}", 'error')
            self.operations['failed'].append(f"Clone error: {str(e)}")
            return False
    
    def copy_file_safe(self, src: Path, dst: Path, description: str) -> bool:
        """Safely copy a file with error handling"""
        try:
            if not src.exists():
                self.operations['skipped'].append(f"{description} (source not found)")
                return False
            
            # Create destination directory if needed
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(src, dst)
            self.operations['success'].append(f"Copied {description}")
            return True
            
        except Exception as e:
            self.operations['failed'].append(f"Copy {description}: {str(e)}")
            return False
    
    def organize_files(self) -> bool:
        """Organize AMF-VIJ files into CSMF structure"""
        self.log("\n📦 Organizing files from AMF-VIJ...", 'header')
        
        if not self.temp_clone_dir.exists():
            self.log("❌ Temp clone directory not found", 'error')
            self.operations['failed'].append("Temp directory missing")
            return False
        
        # Define file mappings: (source, destination, description)
        file_mappings = [
            # Flow implementations to csmf/flows/
            (self.temp_clone_dir / 'amf_vi' / 'flows' / 'base_flow.py',
             self.target_dir / 'csmf' / 'flows' / 'base_flow.py', 'Base flow'),
            (self.temp_clone_dir / 'amf_vi' / 'flows' / 'realnvp.py',
             self.target_dir / 'csmf' / 'flows' / 'realnvp.py', 'RealNVP flow'),
            (self.temp_clone_dir / 'amf_vi' / 'flows' / 'maf.py',
             self.target_dir / 'csmf' / 'flows' / 'maf.py', 'MAF flow'),
            (self.temp_clone_dir / 'amf_vi' / 'flows' / 'rbig.py',
             self.target_dir / 'csmf' / 'flows' / 'rbig.py', 'RBIG flow'),
            
            # Data generators to data/
            (self.temp_clone_dir / 'data' / 'data_generator.py',
             self.target_dir / 'data' / 'data_generator.py', 'Data generator'),
            (self.temp_clone_dir / 'data' / 'visualize_data.py',
             self.target_dir / 'data' / 'visualize_data.py', 'Data visualizer'),
            
            # Utilities to csmf/utils/
            (self.temp_clone_dir / 'amf_vi' / 'kde_kl_divergence.py',
             self.target_dir / 'csmf' / 'utils' / 'metrics.py', 'KL divergence metrics'),
            
            # Tests to tests/
            (self.temp_clone_dir / 'tests' / 'test_flows.py',
             self.target_dir / 'tests' / 'test_flows.py', 'Flow tests'),
            
            # Legacy AMF-VI code
            (self.temp_clone_dir / 'amf_vi' / 'loss.py',
             self.target_dir / 'legacy' / 'amf_vi' / 'loss.py', 'Legacy loss'),
            (self.temp_clone_dir / 'amf_vi' / 'model.py',
             self.target_dir / 'legacy' / 'amf_vi' / 'model.py', 'Legacy model'),
        ]
        
        # Copy files
        success_count = 0
        for src, dst, desc in file_mappings:
            if self.copy_file_safe(src, dst, desc):
                success_count += 1
        
        # Copy all flow files that exist
        flows_dir = self.temp_clone_dir / 'amf_vi' / 'flows'
        if flows_dir.exists():
            for flow_file in flows_dir.glob('*.py'):
                if flow_file.name != '__init__.py':
                    dst = self.target_dir / 'csmf' / 'flows' / flow_file.name
                    self.copy_file_safe(flow_file, dst, f'Flow: {flow_file.stem}')
        
        # Copy main scripts to experiments/
        main_dir = self.temp_clone_dir / 'main'
        if main_dir.exists():
            for script in main_dir.glob('*.py'):
                if script.name != '__init__.py':
                    dst = self.target_dir / 'experiments' / script.name
                    self.copy_file_safe(script, dst, f'Experiment: {script.stem}')
        
        self.log(f"✅ Organized {success_count} core files", 'success')
        return True
    
    def create_config_templates(self) -> bool:
        """Create configuration file templates"""
        self.log("\n⚙️  Creating config templates...", 'header')
        
        configs = {
            'mnist_config.py': '''"""
MNIST Configuration for WP0-WP3
Version: W0.1-MNIST-v1.0
"""

MNIST_CONFIG = {
    # Dataset parameters
    'dataset': {
        'name': 'MNIST',
        'root': './data/mnist',
        'download': True,
        'image_size': (28, 28),
        'num_channels': 1,
    },
    
    # Forward model (inverse problem)
    'forward_model': {
        'type': 'blur_downsample',
        'blur_kernel_size': 5,
        'blur_sigma': 1.0,
        'downsample_factor': 2,
        'noise_std': 0.1,
    },
    
    # Conditioning network (CNN encoder)
    'conditioner': {
        'type': 'cnn',
        'num_layers': 4,
        'channels': [1, 32, 64, 128],
        'kernel_size': 3,
        'activation': 'relu',
        'output_dim': 64,  # h dimension
    },
    
    # FiLM parameters
    'film': {
        'hidden_dims': [64, 64],
        'activation': 'relu',
    },
    
    # Flow architecture
    'flow': {
        'type': 'realnvp',  # or 'maf'
        'num_blocks': 8,
        'hidden_dims': [256, 256],
        'num_experts': 3,
    },
    
    # Training
    'training': {
        'batch_size': 128,
        'learning_rate': 1e-3,
        'num_epochs': 50,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
    },
}
''',
            'sr_config.py': '''"""
Super-Resolution Configuration for WP4.1
Version: W4.1-SR-v1.0
"""

SR_CONFIG = {
    # Dataset parameters
    'dataset': {
        'name': 'DIV2K',
        'root': './data/div2k',
        'patch_size': 64,
        'num_channels': 1,  # Grayscale
    },
    
    # Forward model (SR)
    'forward_model': {
        'type': 'blur_downsample',
        'blur_sigma_range': (0.8, 1.8),
        'downsample_factor': 4,  # 2x or 4x
        'noise_std': 0.05,
    },
    
    # Conditioning network (UNet encoder)
    'conditioner': {
        'type': 'unet_encoder',
        'channels': [1, 32, 64, 128],
        'output_dim': 64,
    },
    
    # Physics-based corrections
    'physics': {
        'use_prox': True,
        'prox_steps': 1,
        'prox_lambda': 0.1,
        'use_pcg': True,
        'pcg_iters': 3,
    },
    
    # Training
    'training': {
        'batch_size': 32,
        'learning_rate': 5e-4,
        'num_epochs': 100,
    },
}
''',
            'sar_config.py': '''"""
SAR Despeckling Configuration for WP4.2
Version: W4.2-SAR-v1.0
"""

SAR_CONFIG = {
    # Dataset parameters
    'dataset': {
        'name': 'SAR_simulated',
        'root': './data/sar',
        'patch_size': 128,
        'num_looks': 4,
    },
    
    # Forward model (multiplicative noise in log domain)
    'forward_model': {
        'type': 'gamma_speckle',
        'num_looks': 4,
        'work_in_log_domain': True,
    },
    
    # Conditioning network
    'conditioner': {
        'type': 'cnn',
        'channels': [1, 32, 64, 128],
        'output_dim': 64,
    },
    
    # Physics (log-domain consistency)
    'physics': {
        'use_log_consistency': True,
        'consistency_weight': 0.1,
    },
    
    # Training
    'training': {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 150,
    },
}
'''
        }
        
        for filename, content in configs.items():
            config_path = self.target_dir / 'configs' / filename
            try:
                config_path.write_text(content)
                self.operations['success'].append(f"Created config: {filename}")
            except Exception as e:
                self.operations['failed'].append(f"Config {filename}: {str(e)}")
        
        self.log(f"✅ Created {len(configs)} config templates", 'success')
        return True
    
    def create_readme_templates(self) -> bool:
        """Create README and documentation templates"""
        self.log("\n📝 Creating documentation...", 'header')
        
        readme_content = f'''# CSMF Project
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

- **Setup date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Target directory:** {self.target_dir.absolute()}
- **In-place mode:** {'Yes' if self.in_place else 'No'}
- **AMF-VIJ source:** {self.repo_url}

## Key References

- W0.1 Dependency Hierarchy: See project documentation
- Chapter 2 Implementation Details: See PDF guides
- Original AMF-VI paper: See legacy/amf_vi/
'''
        
        try:
            readme_path = self.target_dir / 'README.md'
            readme_path.write_text(readme_content)
            self.operations['success'].append("Created README.md")
            self.log(f"✅ Created README.md", 'success')
            return True
        except Exception as e:
            self.operations['failed'].append(f"README creation: {str(e)}")
            return False
    
    def update_requirements(self) -> bool:
        """Create/update requirements.txt"""
        self.log("\n📦 Updating requirements.txt...", 'header')
        
        requirements = '''# CSMF Project Dependencies
# Version: CSMF-v1.0

# Core
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0

# Image processing
Pillow>=9.5.0
opencv-python>=4.7.0
scikit-image>=0.20.0

# Data & visualization
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0

# Testing
pytest>=7.3.0
pytest-cov>=4.1.0

# Utilities
tqdm>=4.65.0
PyYAML>=6.0
tensorboard>=2.13.0

# Optional: For RBIG
scikit-learn>=1.2.0
'''
        
        try:
            req_path = self.target_dir / 'requirements.txt'
            req_path.write_text(requirements)
            self.operations['success'].append("Created requirements.txt")
            self.log(f"✅ Created requirements.txt", 'success')
            return True
        except Exception as e:
            self.operations['failed'].append(f"Requirements: {str(e)}")
            return False
    
    def cleanup(self, keep_temp: bool = False) -> bool:
        """Clean up temporary files"""
        self.log("\n🧹 Cleaning up...", 'header')
        
        if keep_temp:
            self.log(f"   Keeping temp directory: {self.temp_clone_dir}", 'warning')
            self.operations['warnings'].append(f"Temp directory kept: {self.temp_clone_dir}")
            return True
        
        try:
            if self.temp_clone_dir.exists():
                shutil.rmtree(self.temp_clone_dir)
                self.operations['success'].append("Cleaned up temp directory")
                self.log(f"✅ Removed temp directory", 'success')
            return True
        except Exception as e:
            self.operations['warnings'].append(f"Cleanup: {str(e)}")
            self.log(f"⚠️  Could not remove temp directory: {e}", 'warning')
            return True  # Non-critical error
    
    def generate_summary(self) -> Dict:
        """Generate operation summary"""
        return {
            'timestamp': datetime.now().isoformat(),
            'target_directory': str(self.target_dir.absolute()),
            'in_place_mode': self.in_place,
            'statistics': {
                'success': len(self.operations['success']),
                'failed': len(self.operations['failed']),
                'skipped': len(self.operations['skipped']),
                'warnings': len(self.operations['warnings'])
            },
            'operations': self.operations
        }
    
    def print_summary(self, summary: Dict):
        """Print operation summary"""
        self.log("\n" + "="*70, 'header')
        self.log("📊 SETUP SUMMARY", 'header')
        self.log("="*70, 'header')
        
        stats = summary['statistics']
        self.log(f"\n✅ Success: {stats['success']}", 'success')
        self.log(f"⚠️  Skipped: {stats['skipped']}", 'warning')
        self.log(f"❌ Failed:  {stats['failed']}", 'error' if stats['failed'] > 0 else 'info')
        self.log(f"⚠️  Warnings: {stats['warnings']}", 'warning')
        
        if stats['skipped'] > 0:
            self.log("\nSkipped items (expected):", 'warning')
            for item in self.operations['skipped'][:5]:
                self.log(f"  - {item}", 'warning')
            if len(self.operations['skipped']) > 5:
                self.log(f"  ... and {len(self.operations['skipped']) - 5} more", 'warning')
        
        if stats['failed'] > 0:
            self.log("\nFailed items:", 'error')
            for item in self.operations['failed']:
                self.log(f"  - {item}", 'error')
        
        # Save to JSON
        summary_path = self.target_dir / 'setup_summary.json'
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            self.log(f"\n📄 Full summary saved to: {summary_path}", 'info')
        except Exception as e:
            self.log(f"⚠️  Could not save summary: {e}", 'warning')
    
    def run(self, skip_clone: bool = False, no_cleanup: bool = False, auto_confirm: bool = False) -> bool:
        """Run the complete setup process"""
        self.log("\n🚀 Starting CSMF Project Setup", 'header')
        self.log(f"Target directory: {self.target_dir.absolute()}", 'info')
        
        # Safety check for in-place mode
        if self.in_place and not auto_confirm:
            self.log("\n⚠️  IN-PLACE MODE: Setting up in current directory", 'warning')
            existing_files = list(self.target_dir.glob('*'))
            if len(existing_files) > 5:
                self.log(f"   Found {len(existing_files)} existing files/directories", 'warning')
                self.log("   This will merge with existing structure", 'warning')
                response = input("\n   Continue? [y/N]: ").strip().lower()
                if response not in ['y', 'yes']:
                    self.log("\n❌ Setup cancelled by user", 'error')
                    return False
        elif self.in_place and auto_confirm:
            self.log("\n⚠️  IN-PLACE MODE: Auto-confirmed (--yes flag)", 'warning')
        
        steps = [
            ("Creating directory structure", self.create_directory_structure),
            ("Cloning repository", lambda: self.clone_repository() if not skip_clone else True),
            ("Organizing files", self.organize_files),
            ("Creating config templates", self.create_config_templates),
            ("Creating documentation", self.create_readme_templates),
            ("Updating requirements", self.update_requirements),
            ("Cleaning up", lambda: self.cleanup(keep_temp=no_cleanup)),
        ]
        
        for step_name, step_func in steps:
            self.log(f"\n▶️  {step_name}...", 'header')
            try:
                if not step_func():
                    self.log(f"⚠️  {step_name} completed with issues", 'warning')
            except Exception as e:
                self.log(f"❌ {step_name} failed: {e}", 'error')
                self.operations['failed'].append(f"{step_name}: {str(e)}")
        
        # Generate and print summary
        summary = self.generate_summary()
        self.print_summary(summary)
        
        return summary['statistics']['failed'] == 0

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Setup CSMF project from AMF-VIJ repository',
        epilog='Default: Sets up in current directory. Use --target for different location.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--target', default='.', 
                       help='Target directory (default: current directory ".")')
    parser.add_argument('--skip-clone', action='store_true', 
                       help='Skip repository cloning (use existing temp_amfvi_clone)')
    parser.add_argument('--no-cleanup', action='store_true', 
                       help='Keep temporary files for inspection')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt for in-place setup')
    
    args = parser.parse_args()
    
    setup = CSMFSetup(target_dir=args.target)
    
    try:
        success = setup.run(
            skip_clone=args.skip_clone, 
            no_cleanup=args.no_cleanup,
            auto_confirm=args.yes
        )
        
        if success:
            setup.log("\n🎉 CSMF project setup completed successfully!", 'success')
            setup.log(f"\n📍 Next steps:", 'info')
            if args.target == '.':
                setup.log(f"  1. pip install -r requirements.txt", 'info')
                setup.log(f"  2. Start with WP0.1: Implement csmf/conditioning/film.py", 'info')
                setup.log(f"  3. Follow W0.1 dependency hierarchy (see README.md)", 'info')
            else:
                setup.log(f"  1. cd {args.target}", 'info')
                setup.log(f"  2. pip install -r requirements.txt", 'info')
                setup.log(f"  3. Start with WP0.1: Implement csmf/conditioning/film.py", 'info')
            return 0
        else:
            setup.log("\n⚠️  Setup completed with some issues. Check summary above.", 'warning')
            return 1
            
    except KeyboardInterrupt:
        setup.log("\n\n❌ Setup interrupted by user", 'error')
        return 1
    except Exception as e:
        setup.log(f"\n\n❌ Fatal error: {e}", 'error')
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
