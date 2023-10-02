import os
from analyzer import analyze_ballast, analyze_multiple_files
import logging

# Configure logging to print detailed information
logging.basicConfig(level=logging.DEBUG)

def analyze_single_file(file_path):
    result = analyze_ballast(file_path)
    if result:
        print("Filename:", result["Filename"])
        print("Intermediate:", result["Intermediate"])
        print("Shortest:", result["Shortest"])
        print("Longest:", result["Longest"])
        print("Elongation:", result["Elongation"])
        print("Flatness:", result["Flatness"])
        print("Convexity:", result["Convexity"])
        print("Sphericity:", result["Sphericity"])
        print("Roundness:", result["Roundness"])
        print("Roughness:", result["Roughness"])
        print("Sphere Center:", (result["Center X"], result["Center Y"], result["Center Z"]))
        print("Sphere Radius:", result["Radius"])
        print("Aspect Ratio:", result["Aspect Ratio"])
        print("Angularity Index:", result["Angularity Index"])
        print("\n---\n")

def analyze_all_files_in_folder(folder_path, num_cores=None, file_extension=".stl"):
    files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
    files = [os.path.join(folder_path, f) for f in files]
    logging.info(f"Analyzing {len(files)} files in folder: {folder_path}")

    for file in files:
        try:
            results = analyze_multiple_files([file], num_cores)  # Analyzing files one by one
            for result in results:
                if result:
                	print("Filename:", result["Filename"])
                	print("Intermediate:", result["Intermediate"])
                	print("Shortest:", result["Shortest"])
                	print("Longest:", result["Longest"])
                	print("Elongation:", result["Elongation"])
                	print("Flatness:", result["Flatness"])
                	print("Convexity:", result["Convexity"])
                	print("Sphericity:", result["Sphericity"])
                	print("Roundness:", result["Roundness"])
                	print("Roughness:", result["Roughness"])
                	print("Sphere Center:", (result["Center X"], result["Center Y"], result["Center Z"]))
                	print("Sphere Radius:", result["Radius"])
                	print("Aspect Ratio:", result["Aspect Ratio"])
                	print("Angularity Index:", result["Angularity Index"])
                	print("\n---\n")
        except Exception as e:
            logging.error(f"An error occurred while analyzing the file {file}: {e}")
            continue  # Skip the file and continue with the next one
# Analyzing a single file
#single_file_path = "/path/to/your/single/model/file.obj"
#analyze_single_file(single_file_path)

# Analyzing all files in a folder
folder_path = "/home/railcmuvpn/Documents/Morphology_CFC/model"
analyze_all_files_in_folder(folder_path, num_cores=4)
