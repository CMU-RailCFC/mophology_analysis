import os
from ballast_analyzer.analyzer import analyze_ballast

# Use the analyze_ballast function here with the path to your 3D model file
# to run all obj file in the folder please use the first option second option for run single obj file
#option 1
'''
def analyze_all_files_in_folder(folder_path, file_extension=".obj"):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter files by the specified file extension
    files = [f for f in files if f.endswith(file_extension)]

    # Analyze all files with the specified file extension
    for file in files:
        file_path = os.path.join(folder_path, file)
        result = analyze_ballast(file_path)

        # Print the results in the desired format
        print("subjectname:", result["Filename"])
        print("Intermediate:", result["Intermediate"])
        print("Shortest:", result["Shortest"])
        print("Longest:", result["Longest"])
        print("Elongation:", result["Elongation"])
        print("Flatness:", result["Flatness"])
        print("Convexity:", result["Convexity"])
        print("Sphericity:", result["Sphericity"])
        print("Roundness:", result["Roundness"])
        print("Roughness:", result["Roughness"])
        print("Sphere center:", (result["Center X"], result["Center Y"], result["Center Z"]))
        print("Sphere radius:", result["Radius"])
        print("Sphere fit:", ((result["Center X"], result["Center Y"], result["Center Z"]), result["Radius"]))
        print("Angularity Index:", result["Angularity Index"])
        print("Aspect Ratio:", result["Aspect Ratio"])


        print("\n---\n")

folder_path = "path/to/your/folder"
analyze_all_files_in_folder(folder_path)
'''
#option 2
result = analyze_ballast('U15_1/U15_1.stl')
# Print the results in the desired format
print("subjectname:", result["Filename"])
print("Intermediate:", result["Intermediate"])
print("Shortest:", result["Shortest"])
print("Longest:", result["Longest"])
print("Elongation:", result["Elongation"])
print("Flatness:", result["Flatness"])
print("Convexity:", result["Convexity"])
print("Sphericity:", result["Sphericity"])
print("Roundness:", result["Roundness"])
print("Roughness:", result["Roughness"])
print("Sphere center:", (result["Center X"], result["Center Y"], result["Center Z"]))
print("Sphere radius:", result["Radius"])
print("Sphere fit:", ((result["Center X"], result["Center Y"], result["Center Z"]), result["Radius"]))
print("Angularity Index:", result["angularity index"])
print("Aspect Ratio:", result["Aspect Ratio"])
