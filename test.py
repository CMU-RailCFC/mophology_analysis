import os
from analyzer import analyze_ballast, analyze_multiple_files

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

def analyze_all_files_in_folder(folder_path, num_cores=None, file_extension=".obj"):
    files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
    files = [os.path.join(folder_path, f) for f in files]

    results = analyze_multiple_files(files, num_cores)
    for result in results:
        if result:
            print("Filename:", result["Filename"])
            # ... (same as above, printing other properties)
            print("\n---\n")

# Analyzing a single file
single_file_path = "/path/to/your/single/model/file.obj"
analyze_single_file(single_file_path)

# Analyzing all files in a folder
# folder_path = "/path/to/your/models"
# analyze_all_files_in_folder(folder_path, num_cores=4)
