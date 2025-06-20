"""
ENHANCED FAST RUNNER for Robust Roundness Analysis

High-performance batch processing with fixed roundness calculations.
Features:
- Fast parallel processing with WORKING PROGRESS BAR
- No duplicate CSV entries
- Smart overwrite/append options
- Comprehensive error handling
- Streamlined output
- FLEXIBLE FOLDER PATH INPUT
- COMPLETE PARAMETER DISPLAY - Show ALL calculated values!

Save this file as: test.py

DISPLAY MODES:
1. COMPACT - Progress + basic roundness (fast processing)
2. DETAILED - All roundness methods + key geometry  
3. ALL PARAMETERS - Every calculated parameter (complete analysis)
4. SILENT - Progress bar only (fastest)

USAGE METHODS:

1. COMMAND LINE:
   python test.py /path/to/your/stl/files
   python test.py test /path/to/stl/files      (test mode with ALL parameters)

2. CONFIG FILE:
   Create "folder_config.txt" with your folder path
   python test.py

3. INTERACTIVE:
   python test.py
   (Will prompt you to enter folder path and choose display mode)

4. HELP:
   python test.py help

EXAMPLES:
   python test.py /home/user/ballast_stl_files
   python test.py C:\\Users\\User\\STL_Files
   python test.py test                         (shows ALL parameters)
   python test.py

WHAT YOU'LL SEE IN ALL PARAMETERS MODE:
✅ All 5 roundness methods (Wadell, Curvature, Corner, Fourier, Sphere)
✅ Complete geometry (Elongation, Flatness, Convexity, Aspect Ratio)
✅ Physical properties (Volume, Surface Area, Sphere fitting)
✅ Mesh quality (Watertight status, Vertex counts)
✅ Angularity analysis
✅ Method success indicators
"""

import os
import sys
import glob
import time
import logging
import csv
import concurrent.futures
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# Import the fixed analyzer
try:
    from analyzer import analyze_ballast_with_robust_roundness
    print("✅ analyzer loaded - fast processing mode")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Save the fixed analyzer as 'analyzer.py' in same directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(f'analysis_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def save_results_to_csv(results, csv_filename='data_.csv', mode='overwrite'):
    """
    Save all results to CSV at once to prevent duplicates
    
    Args:
        results: List of analysis results
        csv_filename: Output CSV filename
        mode: 'overwrite' or 'append'
    """
    if not results:
        print("❌ No results to save")
        return False
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("❌ No valid results to save")
        return False
    
    try:
        # Get all fieldnames from the first result
        fieldnames = list(valid_results[0].keys())
        
        # Check if file exists for append mode
        file_exists = os.path.isfile(csv_filename)
        
        if mode == 'append' and file_exists:
            # Append mode
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerows(valid_results)
            print(f"➕ Appended {len(valid_results)} results to {csv_filename}")
        else:
            # Overwrite mode (default)
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(valid_results)
            print(f"💾 Saved {len(valid_results)} results to {csv_filename}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error saving CSV: {e}")
        print(f"❌ Error saving CSV: {e}")
        return False

def analyze_single_file_fast(filename):
    """Fast single file analysis without debug output"""
    try:
        return analyze_ballast_with_robust_roundness(
            filename, 
            roundness_method='comprehensive',
            enable_debug=False  # Disable debug for speed
        )
    except Exception as e:
        logging.error(f"Error analyzing {filename}: {e}")
        return None

def print_analysis_summary(result, show_details=False, show_all_params=False):
    """Print concise analysis results with option to show all parameters"""
    if not result:
        return
    
    filename = os.path.basename(result.get('Filename', 'Unknown'))
    vertices = result.get('Final Vertex Count', 0)
    roundness = result.get('Roundness', 0)
    quality = result.get('Roundness_Quality', 0)
    working = result.get('Working_Methods', 0)
    
    if show_all_params:
        # Show ALL parameters in detail
        print(f"\n📂 {filename}")
        print(f"{'='*60}")
        
        # Basic info
        print(f"🔢 MESH INFO:")
        print(f"   Original Vertices: {result.get('Original Vertex Count', 'N/A'):,}")
        print(f"   Final Vertices: {vertices:,}")
        print(f"   Faces: {result.get('Number of Faces', 'N/A'):,}")
        print(f"   Reduction: {result.get('Vertex Reduction %', 0):.1f}%")
        
        # Geometric properties
        print(f"\n📐 GEOMETRIC PROPERTIES:")
        print(f"   Shortest: {result.get('Shortest', 0):.4f}")
        print(f"   Intermediate: {result.get('Intermediate', 0):.4f}")
        print(f"   Longest: {result.get('Longest', 0):.4f}")
        print(f"   Elongation: {result.get('Elongation', 0):.4f}")
        print(f"   Flatness: {result.get('Flatness', 0):.4f}")
        print(f"   Aspect Ratio: {result.get('Aspect Ratio', 0):.4f}")
        
        # Shape properties
        print(f"\n🔮 SHAPE PROPERTIES:")
        print(f"   Convexity: {result.get('Convexity', 0):.4f}")
        print(f"   Sphericity: {result.get('Sphericity', 0):.4f}")
        print(f"   Sphericity2: {result.get('Sphericity2', 0):.4f}")
        print(f"   Roughness: {result.get('Roughness', 0):.6f}")
        
        # ALL ROUNDNESS METHODS
        print(f"\n🎯 ROUNDNESS ANALYSIS (DETAILED):")
        print(f"   🏆 COMPOSITE: {roundness:.4f} (Method: {result.get('Roundness_Method', 'N/A')})")
        print(f"   ⭐ Quality Score: {quality:.2f} | Working Methods: {working}/5")
        print(f"\n   📊 INDIVIDUAL METHODS:")
        wadell = result.get('Roundness_Wadell', 0)
        curvature = result.get('Roundness_Curvature', 0)
        corner = result.get('Roundness_Corner', 0)
        fourier = result.get('Roundness_Fourier', 0)
        sphere = result.get('Roundness_Sphere', 0)
        
        print(f"   🔴 Wadell-style:    {wadell:.4f} {'✅' if wadell > 0 else '❌'}")
        print(f"   🟠 Curvature-based: {curvature:.4f} {'✅' if curvature > 0 else '❌'}")
        print(f"   🟡 Corner detection:{corner:.4f} {'✅' if corner > 0 else '❌'}")
        print(f"   🟢 Fourier-based:   {fourier:.4f} {'✅' if fourier > 0 else '❌'}")
        print(f"   🔵 Sphere-based:    {sphere:.4f} {'✅' if sphere > 0 else '❌'}")
        print(f"   🔶 Est. Corners:    {result.get('Estimated_Corner_Count', 0)}")
        
        # Angularity
        print(f"\n📐 ANGULARITY:")
        print(f"   Angularity Index: {result.get('Angularity Index', 0):.4f}")
        print(f"   Normalized Angularity: {result.get('Normalized Angularity Index', 0):.4f}")
        
        # Physical properties
        print(f"\n🏗️  PHYSICAL PROPERTIES:")
        print(f"   Surface Area: {result.get('Surface Area', 0):.4f}")
        print(f"   Volume: {result.get('Volume', 0):.4f}")
        print(f"   Convex Hull Faces: {result.get('Convex Hull Faces', 'N/A')}")
        print(f"   Convex Hull Vertices: {result.get('Convex Hull Vertices', 'N/A')}")
        
        # Sphere fitting
        center_x = result.get('Center X', 0)
        center_y = result.get('Center Y', 0)
        center_z = result.get('Center Z', 0)
        radius = result.get('Radius', 0)
        print(f"\n⚪ FITTED SPHERE:")
        print(f"   Center: ({center_x:.4f}, {center_y:.4f}, {center_z:.4f})")
        print(f"   Radius: {radius:.4f}")
        
        # Mesh quality
        print(f"\n🌊 MESH QUALITY:")
        print(f"   Watertight: {result.get('Is Watertight', 'Unknown')}")
        print(f"   Valid: {result.get('Is Valid', 'Unknown')}")
        
    elif show_details:
        # Standard detailed view
        wadell = result.get('Roundness_Wadell', 0)
        curvature = result.get('Roundness_Curvature', 0)
        corner = result.get('Roundness_Corner', 0)
        fourier = result.get('Roundness_Fourier', 0)
        sphere = result.get('Roundness_Sphere', 0)
        
        print(f"\n📂 {filename}")
        print(f"   Vertices: {vertices:,}, Roundness: {roundness:.3f}, Quality: {quality:.2f}")
        print(f"   Roundness Methods: W:{wadell:.3f} C:{curvature:.3f} Co:{corner:.3f} F:{fourier:.3f} S:{sphere:.3f}")
        print(f"   Geometry: Elongation:{result.get('Elongation', 0):.3f}, Flatness:{result.get('Flatness', 0):.3f}, Convexity:{result.get('Convexity', 0):.3f}")
    else:
        # Compact view with more info
        status = "✅" if working >= 3 else "⚠️" if working >= 1 else "❌"
        wadell = result.get('Roundness_Wadell', 0)
        sphere = result.get('Roundness_Sphere', 0)
        convex = result.get('Convexity', 0)
        print(f"{status} {filename}: R={roundness:.3f} W={wadell:.3f} S={sphere:.3f} Q={quality:.2f} Conv={convex:.3f} Methods={working}/5")

def get_folder_path():
    """Get folder path from command line, config, or user input"""
    import sys
    
    # Method 1: Command line argument
    if len(sys.argv) > 1 and not sys.argv[1] in ['test', 'help']:
        folder_path = sys.argv[1]
        if os.path.exists(folder_path):
            print(f"📁 Using folder from command line: {folder_path}")
            return folder_path
        else:
            print(f"❌ Folder not found: {folder_path}")
    
    # Method 2: Check for config file
    config_file = 'folder_config.txt'
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                folder_path = f.read().strip()
            if os.path.exists(folder_path):
                print(f"📁 Using folder from config file: {folder_path}")
                return folder_path
            else:
                print(f"⚠️  Config file exists but folder not found: {folder_path}")
        except Exception as e:
            print(f"⚠️  Error reading config file: {e}")
    
    # Method 3: Interactive input
    print("\n" + "="*60)
    print("📁 FOLDER PATH CONFIGURATION")
    print("="*60)
    print("Enter the path to your STL files folder:")
    print("Examples:")
    print("  /home/user/stl_files")
    print("  C:\\Users\\User\\Documents\\STL_Files")
    print("  .")
    print("")
    
    while True:
        try:
            folder_path = input("📂 Folder path: ").strip()
            
            # Handle empty input
            if not folder_path:
                print("❌ Please enter a folder path")
                continue
            
            # Expand user path (~)
            folder_path = os.path.expanduser(folder_path)
            
            # Convert to absolute path
            folder_path = os.path.abspath(folder_path)
            
            # Check if folder exists
            if os.path.exists(folder_path):
                # Check if it contains STL files
                stl_files = glob.glob(os.path.join(folder_path, "*.stl"))
                if stl_files:
                    print(f"✅ Found {len(stl_files)} STL files in: {folder_path}")
                    
                    # Save to config file for next time
                    try:
                        with open(config_file, 'w') as f:
                            f.write(folder_path)
                        print(f"💾 Saved folder path to {config_file} for next time")
                    except:
                        pass  # Don't fail if we can't save config
                    
                    return folder_path
                else:
                    print(f"⚠️  Folder exists but no STL files found: {folder_path}")
                    choice = input("Continue anyway? (y/n): ").lower().strip()
                    if choice == 'y':
                        return folder_path
            else:
                print(f"❌ Folder not found: {folder_path}")
                print("Please check the path and try again")
                
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")

def analyze_folder_fast(folder_path, num_cores=4, file_extension=".stl", max_files=None, show_details=False, show_all_params=False, csv_mode='overwrite'):
    """Fast folder analysis with parallel processing and WORKING PROGRESS BAR"""
    print(f"\n🗂️  FAST ANALYSIS: {folder_path}")
    print(f"Cores: {num_cores}, Extension: {file_extension}")
    
    if show_all_params:
        print("📊 Display mode: ALL PARAMETERS")
    elif show_details:
        print("📊 Display mode: DETAILED")
    else:
        print("📊 Display mode: COMPACT")
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return []
    
    # Find files
    files = glob.glob(os.path.join(folder_path, f"*{file_extension}"))
    
    if not files:
        print(f"❌ No {file_extension} files found")
        return []
    
    if max_files:
        files = files[:max_files]
        print(f"📊 Processing {len(files)} files (limited)")
    else:
        print(f"📊 Processing {len(files)} files")
    
    start_time = time.time()
    results = []
    successful = 0
    failed = 0
    
    try:
        if num_cores > 1 and len(files) > 3:
            # Parallel processing with WORKING progress tracking
            print("🚀 Using parallel processing with real-time progress...")
            print(f"📊 Processing {len(files)} files with {num_cores} cores")
            
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                # Submit all jobs
                future_to_file = {executor.submit(analyze_single_file_fast, file_path): file_path 
                                 for file_path in files}
                
                # Progress tracking variables
                completed = 0
                start_progress_time = time.time()
                
                # Process results as they complete with progress
                for future in concurrent.futures.as_completed(future_to_file):
                    completed += 1
                    file_path = future_to_file[future]
                    filename = os.path.basename(file_path)
                    
                    # Calculate progress, ETA, and throughput
                    progress_pct = (completed / len(files)) * 100
                    elapsed = time.time() - start_progress_time
                    
                    if completed > 0 and elapsed > 0:
                        avg_time_per_file = elapsed / completed
                        remaining_files = len(files) - completed
                        eta_seconds = remaining_files * avg_time_per_file
                        eta_str = f"{eta_seconds/60:.1f}m" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
                        
                        # Current throughput
                        files_per_min = (completed / elapsed) * 60
                        throughput_str = f"{files_per_min:.1f}f/min"
                    else:
                        eta_str = "calculating..."
                        throughput_str = "calculating..."
                    
                    # Progress bar (20 characters wide)
                    filled_length = int(progress_pct // 5)
                    progress_bar = "█" * filled_length + "▒" * (20 - filled_length)
                    
                    # Compact status line with throughput
                    status_line = f"[{progress_bar}] {completed}/{len(files)} ({progress_pct:.1f}%) | {throughput_str} | ETA: {eta_str} | {filename[:20]}"
                    print(f"\r{status_line:<85}", end="", flush=True)
                    
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            successful += 1
                            if show_details or show_all_params:
                                print()  # New line for detailed output
                                print_analysis_summary(result, show_details, show_all_params)
                        else:
                            failed += 1
                            if show_details or show_all_params:
                                print()
                                print(f"❌ {filename}: Failed")
                    except Exception as e:
                        failed += 1
                        if show_details or show_all_params:
                            print()
                            print(f"❌ {filename}: Error - {str(e)[:50]}...")
                        logging.error(f"Error processing {filename}: {e}")
                
                # Clear progress line and show final completion stats
                total_time = time.time() - start_progress_time
                final_throughput = (len(files) / total_time) * 60 if total_time > 0 else 0
                print(f"\r🎉 Parallel processing complete! {successful} successful, {failed} failed")
                print(f"   ⚡ Final speed: {final_throughput:.1f} files/min | Total time: {total_time:.1f}s" + " " * 10)
                
        else:
            # Sequential processing with progress
            print("🔄 Using sequential processing with progress...")
            start_seq_time = time.time()
            
            for i, file_path in enumerate(files, 1):
                filename = os.path.basename(file_path)
                progress_pct = (i / len(files)) * 100
                
                # Calculate ETA
                if i > 1:
                    elapsed = time.time() - start_seq_time
                    avg_time = elapsed / (i - 1)
                    remaining = len(files) - i
                    eta_seconds = remaining * avg_time
                    eta_str = f"{eta_seconds/60:.1f}m" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
                else:
                    eta_str = "calculating..."
                
                # Progress bar for sequential
                filled_length = int(progress_pct // 5)
                progress_bar = "█" * filled_length + "▒" * (20 - filled_length)
                
                print(f"[{progress_bar}] {i}/{len(files)} ({progress_pct:.1f}%) | ETA: {eta_str} | {filename}")
                
                result = analyze_single_file_fast(file_path)
                if result:
                    results.append(result)
                    successful += 1
                    if show_all_params or show_details:
                        print_analysis_summary(result, show_details, show_all_params)
                    else:
                        print_analysis_summary(result, False, False)
                else:
                    failed += 1
                    print(f"   ❌ Analysis failed")
            
            print(f"✅ Sequential processing complete!")
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
    except Exception as e:
        logging.error(f"Critical error: {e}")
        print(f"❌ Critical error: {e}")
    
    elapsed_time = time.time() - start_time
    
    # Summary with enhanced performance metrics
    print(f"\n{'='*60}")
    print(f"📊 FAST ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {len(files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/len(files)*100):.1f}%" if files else "0%")
    print(f"Total time: {elapsed_time:.1f}s")
    print(f"Average per file: {elapsed_time/len(files):.1f}s" if files else "N/A")
    
    # Processing speed metrics
    if elapsed_time > 0:
        files_per_minute = (len(files) / elapsed_time) * 60
        files_per_hour = files_per_minute * 60
        print(f"Processing speed: {files_per_minute:.1f} files/min ({files_per_hour:.0f} files/hour)")
    
    # Core utilization info
    if num_cores > 1 and len(files) > 3:
        theoretical_speedup = min(num_cores, len(files))
        print(f"Cores used: {num_cores} (theoretical max speedup: {theoretical_speedup:.1f}x)")
    
    if successful > 0:
        # Save all results to CSV at once (prevents duplicates)
        csv_saved = save_results_to_csv(results, 'data_.csv', mode=csv_mode)
        
        # Quick statistics
        working_methods = [r.get('Working_Methods', 0) for r in results]
        roundness_values = [r.get('Roundness', 0) for r in results if r.get('Roundness', 0) > 0]
        
        if working_methods:
            avg_working = sum(working_methods) / len(working_methods)
            print(f"Average working methods: {avg_working:.1f}/5")
        
        if roundness_values:
            avg_roundness = sum(roundness_values) / len(roundness_values)
            print(f"Average roundness: {avg_roundness:.3f}")
            print(f"Roundness range: {min(roundness_values):.3f} - {max(roundness_values):.3f}")
        
        if csv_saved:
            print(f"📁 Results saved to: data_.csv")
        else:
            print(f"⚠️  CSV save failed - check log for details")
    
    return results

def show_help():
    """Show help information"""
    print("="*60)
    print("BALLAST ANALYZER - HELP")
    print("="*60)
    print("USAGE:")
    print("  python test.py [folder_path]           # Process STL files")
    print("  python test.py test [folder_path]      # Test mode (few files)")
    print("  python test.py help                    # Show this help")
    print("")
    print("EXAMPLES:")
    print("  python test.py /home/user/stl_files")
    print("  python test.py test")
    print("  python test.py")
    print("")
    print("FOLDER PATH OPTIONS:")
    print("  1. Command line: python test.py /path/to/folder")
    print("  2. Config file: Create 'folder_config.txt' with folder path")
    print("  3. Interactive: Run without arguments, enter path when prompted")
    print("")
    print("OUTPUT:")
    print("  - Results saved to: data_.csv")
    print("  - Log saved to: analysis_log_YYYYMMDD_HHMMSS.log")
    print("="*60)

def main():
    """Fast main execution with flexible folder path input"""
    print("="*60)
    print("BALLAST ANALYZER")
    print("="*60)
    print("✅ roundness calculations")
    print("🚀 High-speed parallel processing")
    print("📊 Comprehensive analysis output")
    print("🔧 NO DUPLICATE CSV ENTRIES")
    print("📁 FLEXIBLE FOLDER PATH INPUT")
    print("="*60)
    
    # Get folder path using flexible method
    folder_path = get_folder_path()
    
    # Processing settings
    num_cores = 4              # CPU cores for parallel processing
    max_files = None           # None = process all files, or set number for testing
    show_details = False       # True = detailed output per file
    csv_mode = 'overwrite'     # 'overwrite' or 'append' - prevents duplicates by default
    
    # Optional: Test single file first (set to True for debugging)
    test_single_file = False
    
    # Check if CSV exists and ask about overwrite
    csv_filename = 'data_.csv'
    if os.path.exists(csv_filename) and csv_mode == 'overwrite':
        print(f"⚠️  CSV file '{csv_filename}' already exists")
        print("Choose mode: [o]verwrite (default), [a]ppend, or [q]uit")
        try:
            choice = input("Mode (o/a/q): ").lower().strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            return
        
        if choice == 'q':
            print("Exiting...")
            return
        elif choice == 'a':
            csv_mode = 'append'
            print("📝 Will append to existing CSV")
        else:
            print("📝 Will overwrite existing CSV")
    
    if test_single_file:
        # Quick single file test
        test_files = glob.glob(os.path.join(folder_path, "*.stl"))
        if test_files:
            print("🧪 Quick single file test...")
            single_result = analyze_single_file_fast(test_files[0])
            if single_result:
                print_analysis_summary(single_result, show_details=True)
                working = single_result.get('Working_Methods', 0)
                print(f"✅ Test passed: {working}/5 methods working")
            else:
                print("❌ Test failed")
                return
    
    # Main processing
    try:
        results = analyze_folder_fast(
            folder_path,
            num_cores=num_cores,
            file_extension=".stl",
            max_files=max_files,
            show_details=show_details,
            csv_mode=csv_mode
        )
        
        if results:
            successful = len([r for r in results if r])
            print(f"\n🎉 PROCESSING COMPLETE!")
            print(f"✅ Successfully processed {successful} files")
            
            # Check fix effectiveness
            working_methods = [r.get('Working_Methods', 0) for r in results if r]
            if working_methods:
                avg_working = sum(working_methods) / len(working_methods)
                
                if avg_working >= 4:
                    print(f"🎊 EXCELLENT: {avg_working:.1f}/5 roundness methods working!")
                elif avg_working >= 3:
                    print(f"✅ GOOD: {avg_working:.1f}/5 roundness methods working")
                elif avg_working >= 1:
                    print(f"⚠️  PARTIAL: {avg_working:.1f}/5 roundness methods working")
                else:
                    print(f"❌ POOR: {avg_working:.1f}/5 roundness methods working")
            
            print(f"📁 Final results in: {csv_filename}")
            
            # Show brief statistics
            if working_methods:
                roundness_values = [r.get('Roundness', 0) for r in results if r and r.get('Roundness', 0) > 0]
                if roundness_values:
                    print(f"\n📈 ROUNDNESS STATISTICS:")
                    print(f"   Average: {sum(roundness_values)/len(roundness_values):.3f}")
                    print(f"   Range: {min(roundness_values):.3f} - {max(roundness_values):.3f}")
                    
                # Show breakdown by method success
                wadell_success = len([r for r in results if r and r.get('Roundness_Wadell', 0) > 0])
                curvature_success = len([r for r in results if r and r.get('Roundness_Curvature', 0) > 0])
                corner_success = len([r for r in results if r and r.get('Roundness_Corner', 0) > 0])
                fourier_success = len([r for r in results if r and r.get('Roundness_Fourier', 0) > 0])
                sphere_success = len([r for r in results if r and r.get('Roundness_Sphere', 0) > 0])
                
                print(f"\n🔧 METHOD SUCCESS RATES:")
                print(f"   Wadell: {wadell_success}/{successful} ({wadell_success/successful*100:.0f}%)")
                print(f"   Curvature: {curvature_success}/{successful} ({curvature_success/successful*100:.0f}%)")
                print(f"   Corner: {corner_success}/{successful} ({corner_success/successful*100:.0f}%)")
                print(f"   Fourier: {fourier_success}/{successful} ({fourier_success/successful*100:.0f}%)")
                print(f"   Sphere: {sphere_success}/{successful} ({sphere_success/successful*100:.0f}%)")
        else:
            print("❌ No files processed successfully")
            
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    except Exception as e:
        logging.error(f"Main processing error: {e}")
        print(f"❌ Error: {e}")

def quick_test_mode():
    """Quick test mode for debugging - processes just a few files with ALL parameters displayed"""
    print("="*60)
    print("🧪 QUICK TEST MODE - ALL PARAMETERS")
    print("="*60)
    
    # Get folder path using flexible method
    folder_path = get_folder_path()
    
    # Test settings
    test_files_count = 3
    enable_debug = True
    
    # Find test files
    test_files = glob.glob(os.path.join(folder_path, "*.stl"))[:test_files_count]
    
    if not test_files:
        print(f"❌ No STL files found in {folder_path}")
        return
    
    print(f"Testing {len(test_files)} files with FULL debug output and ALL parameters...")
    print("📊 This will show every calculated parameter for testing purposes")
    
    results = []
    for i, file_path in enumerate(test_files, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(test_files)}] TESTING: {os.path.basename(file_path)}")
        print(f"{'='*80}")
        
        try:
            result = analyze_ballast_with_robust_roundness(
                file_path, 
                roundness_method='comprehensive',
                enable_debug=enable_debug  # Show detailed debug output
            )
            
            if result:
                results.append(result)
                # Show ALL parameters in test mode
                print_analysis_summary(result, show_details=True, show_all_params=True)
                
                # Additional test-specific info
                print(f"\n🧪 TEST ANALYSIS:")
                working = result.get('Working_Methods', 0)
                quality = result.get('Roundness_Quality', 0)
                
                if working >= 4:
                    print(f"   🎉 EXCELLENT: {working}/5 methods working, Quality: {quality:.2f}")
                elif working >= 3:
                    print(f"   ✅ GOOD: {working}/5 methods working, Quality: {quality:.2f}")
                elif working >= 1:
                    print(f"   ⚠️  PARTIAL: {working}/5 methods working, Quality: {quality:.2f}")
                else:
                    print(f"   ❌ POOR: Only {working}/5 methods working, Quality: {quality:.2f}")
                    
            else:
                print("❌ Analysis failed")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        print(f"\n{'='*80}")
        print(f"🧪 TEST SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Test completed: {len(results)}/{len(test_files)} successful")
        
        # Test-specific statistics
        if len(results) > 1:
            avg_working = sum(r.get('Working_Methods', 0) for r in results) / len(results)
            avg_roundness = sum(r.get('Roundness', 0) for r in results) / len(results)
            avg_quality = sum(r.get('Roundness_Quality', 0) for r in results) / len(results)
            
            print(f"📊 AVERAGE PERFORMANCE:")
            print(f"   Working Methods: {avg_working:.1f}/5")
            print(f"   Roundness: {avg_roundness:.3f}")
            print(f"   Quality Score: {avg_quality:.2f}")
        
        # Save test results
        if save_results_to_csv(results, 'test_results.csv', mode='overwrite'):
            print("📁 Test results saved to: test_results.csv")
            
        print(f"\n💡 If test results look good, run main analysis:")
        print(f"   python test.py {folder_path}")
    else:
        print("❌ All tests failed - check your analyzer.py file")

if __name__ == "__main__":
    import sys
    
    # Check for help
    if len(sys.argv) > 1 and sys.argv[1] == 'help':
        show_help()
        sys.exit(0)
    
    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        quick_test_mode()
    else:
        main()
