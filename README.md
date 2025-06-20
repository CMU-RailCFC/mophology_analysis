# Morphology analyzer V2.1
# Ballast Analyzer Runner - User Manual

## Table of Contents
- [Overview](#overview)
- [Installation & Setup](#installation--setup)
- [Quick Start](#quick-start)
- [Usage Methods](#usage-methods)
- [Display Modes](#display-modes)
- [Configuration Options](#configuration-options)
- [Performance Settings](#performance-settings)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)
- [Advanced Usage](#advanced-usage)

## Overview

The Ballast Analyzer Runner (`test.py`) is a high-performance batch processing tool for analyzing 3D ballast particle meshes. It provides:

- **Fast parallel processing** with real-time progress tracking
- **Multiple display modes** from compact to comprehensive parameter output  
- **Flexible folder path input** with no code editing required
- **Robust error handling** and logging
- **No duplicate CSV entries** with smart append/overwrite options
- **Comprehensive roundness analysis** using 5 different methods

## Installation & Setup

### Prerequisites
```bash
# Required Python packages
pip install trimesh numpy scipy scikit-learn

# Optional but recommended
pip install pandas matplotlib  # For data analysis
```

### File Structure
```
your_project/
├── test.py           # Runner script (this file)
├── analyzer.py       # Analyzer engine (required)
├── folder_config.txt # Optional: folder path config
├── data_.csv         # Output: analysis results
└── analysis_log_*.log # Output: processing logs
```

### Setup
1. Save `test.py` and `analyzer.py` in the same directory
2. Ensure your STL files are in a single folder
3. Run the script - no configuration editing required!

## Quick Start

### Simplest Usage
```bash
python test.py /path/to/your/stl/files
```

### Interactive Mode
```bash
python test.py
# Will prompt for folder path and display options
```

### Test Mode (3 files with full details)
```bash
python test.py test
```

## Usage Methods

### 1. Command Line (Recommended)
Direct folder specification:
```bash
python test.py /home/user/ballast_stl_files
python test.py C:\Users\User\STL_Files
python test.py .  # Current directory
```

### 2. Config File
Create `folder_config.txt` with your folder path:
```
/home/user/ballast_stl_files
```
Then run:
```bash
python test.py
```

### 3. Interactive Input
Run without arguments and follow prompts:
```bash
python test.py
📂 Folder path: /your/path/here
```

### 4. Help
```bash
python test.py help
```

## Display Modes

Choose how much detail to see during processing:

### Mode 1: COMPACT (Default)
**Best for**: Fast processing, large batches
```
✅ ballast_001.stl: R=0.724 W=0.681 S=0.789 Q=0.8 Conv=0.912 Methods=4/5
✅ ballast_002.stl: R=0.693 W=0.712 S=0.734 Q=0.6 Conv=0.876 Methods=3/5
```

### Mode 2: DETAILED  
**Best for**: Monitoring roundness methods
```
📂 ballast_001.stl
   Vertices: 12,450, Roundness: 0.724, Quality: 0.80
   Roundness Methods: W:0.681 C:0.734 Co:0.756 F:0.692 S:0.789
   Geometry: Elongation:0.823, Flatness:0.697, Convexity:0.912
```

### Mode 3: ALL PARAMETERS
**Best for**: Research, detailed analysis, debugging
```
📂 ballast_001.stl
============================================================
🔢 MESH INFO:
   Original Vertices: 15,234
   Final Vertices: 12,450
   Faces: 24,896
   Reduction: 18.3%

📐 GEOMETRIC PROPERTIES:
   Shortest: 12.4567
   Intermediate: 15.2341
   Longest: 18.7654
   Elongation: 0.8234
   Flatness: 0.6970
   Aspect Ratio: 0.6634

🎯 ROUNDNESS ANALYSIS (DETAILED):
   🏆 COMPOSITE: 0.7240 (Method: comprehensive)
   ⭐ Quality Score: 0.80 | Working Methods: 4/5
   
   📊 INDIVIDUAL METHODS:
   🔴 Wadell-style:    0.6810 ✅
   🟠 Curvature-based: 0.7340 ✅
   🟡 Corner detection:0.7560 ✅
   🟢 Fourier-based:   0.6920 ✅
   🔵 Sphere-based:    0.7890 ✅

[... and much more detail ...]
```

### Mode 4: SILENT
**Best for**: Maximum speed, minimal output
```
[████████████████████] 150/150 (100.0%) | 15.2f/min | ETA: 0s | ballast_150.stl
🎉 Processing complete! 148 successful, 2 failed
```

## Configuration Options

### Parallel Processing
- **Default**: 4 CPU cores
- **Auto-adjustment**: Reduces cores in ALL PARAMETERS mode for readability
- **Manual**: Edit `num_cores` in script or use interactive mode

### File Filtering
- **Extension**: `.stl` files (configurable)
- **Limit**: Process subset with `max_files` setting
- **Pattern**: Supports glob patterns

### CSV Output
- **Overwrite**: Replace existing CSV (default)
- **Append**: Add to existing CSV
- **Interactive**: Prompts when file exists

## Performance Settings

### Recommended Settings by Use Case

#### Research/Analysis (Quality Priority)
```
Display Mode: ALL PARAMETERS
Cores: 2-4 (for readability)
File Limit: None
Show Details: True
```

#### Production/Batch (Speed Priority)  
```
Display Mode: COMPACT or SILENT
Cores: 4-8 (maximum available)
File Limit: None
Show Details: False
```

#### Testing/Debugging
```
Display Mode: ALL PARAMETERS
Cores: 1-2
File Limit: 3-10 files
Show Details: True
```

### Performance Metrics
The system reports:
- **Files per minute** (real-time throughput)
- **ETA** (estimated time remaining)
- **Core utilization** (theoretical vs actual speedup)
- **Success rate** (percentage of files processed successfully)

## Output Files

### Primary Output: `data_.csv`
Contains all calculated parameters for each STL file:

**Roundness Parameters:**
- `Roundness` - Composite roundness score
- `Roundness_Wadell` - Wadell-style roundness
- `Roundness_Curvature` - Curvature-based roundness
- `Roundness_Corner` - Corner detection roundness
- `Roundness_Fourier` - Fourier-based roundness
- `Roundness_Sphere` - Sphere-based roundness
- `Roundness_Quality` - Quality score (0-1)
- `Working_Methods` - Number of successful methods (0-5)

**Geometric Parameters:**
- `Shortest`, `Intermediate`, `Longest` - Principal dimensions
- `Elongation`, `Flatness`, `Aspect Ratio` - Shape ratios
- `Convexity`, `Sphericity`, `Sphericity2` - Form factors
- `Roughness` - Surface roughness measure

**Physical Properties:**
- `Surface Area`, `Volume` - Basic measurements
- `Center X/Y/Z`, `Radius` - Fitted sphere parameters
- `Angularity Index` - Angular characteristics

**Mesh Information:**
- `Original/Final Vertex Count` - Mesh resolution
- `Number of Faces` - Mesh complexity
- `Is Watertight` - Mesh quality indicator

### Log Files: `analysis_log_YYYYMMDD_HHMMSS.log`
Contains detailed processing information:
- File processing status
- Error messages and stack traces
- Performance metrics
- System information

### Test Output: `test_results.csv`
Generated in test mode with sample results for validation.

## Troubleshooting

### Common Issues

#### "No STL files found"
**Problem**: Script can't find STL files in specified folder
**Solutions**:
- Check folder path spelling
- Ensure files have `.stl` extension
- Try absolute path instead of relative
- Check file permissions

#### "Import error: analyzer"
**Problem**: Missing or incorrect analyzer.py file
**Solutions**:
- Ensure `analyzer.py` is in same directory as `test.py`
- Check file permissions
- Verify analyzer.py syntax

#### "Memory error" or crashes
**Problem**: Files too large or too many for available RAM
**Solutions**:
- Reduce number of cores: `num_cores = 2`
- Process files in smaller batches: `max_files = 50`
- Use SILENT mode to reduce memory usage
- Close other applications

#### Slow processing
**Problem**: Lower than expected throughput
**Solutions**:
- Use COMPACT or SILENT mode
- Increase number of cores
- Check if files are unusually large/complex
- Monitor CPU and memory usage

#### Progress bar not updating
**Problem**: Progress appears frozen
**Solutions**:
- Large files may take time - wait longer
- Check log files for error messages
- Try sequential processing (reduce cores to 1)
- Restart with smaller test batch

### Debug Mode
```bash
python test.py test  # Processes 3 files with full debug output
```

### Logging
Check log files for detailed error information:
```bash
tail -f analysis_log_*.log  # Monitor real-time
grep ERROR analysis_log_*.log  # Find errors
```

## Examples

### Basic Usage Examples

#### Process all STL files in a folder
```bash
python test.py /home/user/ballast_data
```

#### Process with detailed output
```bash
python test.py /home/user/ballast_data
# Choose option 2 (DETAILED) when prompted
```

#### Test mode with full parameters
```bash
python test.py test /home/user/ballast_data
```

#### Append to existing results
```bash
python test.py /home/user/new_ballast_data
# Choose 'a' (append) when prompted about existing CSV
```

### Advanced Usage Examples

#### Process subset for testing
Edit script to set:
```python
max_files = 10  # Process only first 10 files
```

#### Custom core configuration
Edit script to set:
```python
num_cores = 8  # Use 8 CPU cores
```

#### Silent processing for automation
```bash
python test.py /data/ballast_files
# Choose option 4 (SILENT) for minimal output
```

### Batch Processing Multiple Folders
```bash
# Process multiple folders sequentially
for folder in folder1 folder2 folder3; do
    python test.py $folder
done
```

## Advanced Usage

### Integration with Other Tools

#### Export to Excel
```python
import pandas as pd
df = pd.read_csv('data_.csv')
df.to_excel('ballast_analysis.xlsx', index=False)
```

#### Statistical Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data_.csv')
print(df['Roundness'].describe())
plt.hist(df['Roundness'], bins=50)
plt.show()
```

#### Quality Filtering
```python
import pandas as pd
df = pd.read_csv('data_.csv')
# Keep only high-quality results
high_quality = df[df['Roundness_Quality'] > 0.7]
```

### Automation Scripts

#### Automated Processing Pipeline
```bash
#!/bin/bash
# Process all folders in a directory
for folder in /data/ballast_*/; do
    if [ -d "$folder" ]; then
        echo "Processing $folder"
        python test.py "$folder"
        # Move results to archive
        mv data_.csv "results/$(basename $folder)_results.csv"
    fi
done
```

#### Monitor Processing
```bash
# Monitor progress and log errors
python test.py /data/ballast_files 2>&1 | tee processing.log
```

### Performance Optimization

#### For Large Datasets (>1000 files)
```python
# Recommended settings in script
num_cores = 8  # Use more cores
show_details = False  # Minimal output
csv_mode = 'overwrite'  # Fresh start
```

#### For High Accuracy Analysis
```python
# Settings for research quality
num_cores = 2  # Reduce for stability
show_all_params = True  # Full detail
enable_debug = True  # Maximum information
```

#### Memory-Constrained Systems
```python
# Conservative settings
num_cores = 2  # Reduce parallel load
max_files = 25  # Process in smaller batches
show_details = False  # Minimize memory usage
```

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log files for error details
3. Test with a small subset of files first
4. Ensure all dependencies are installed correctly

**Log Location**: `analysis_log_YYYYMMDD_HHMMSS.log`
**Results Location**: `data_.csv`
