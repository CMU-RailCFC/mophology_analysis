import os
import logging
from analyzer import BallastAnalyzer

# Configure logging to print detailed information
logging.basicConfig(level=logging.DEBUG)

def analyze_single_file(analyzer, file_path):
    result = analyzer.analyze_ballast(file_path)
    if result:
        for key, value in result.items():
            print(f"{key}: {value}")
            
            # Print PCA results
            if "Principal Axis" in key:
                print(f"{key}: {value}")

        print("\n---\n")

def analyze_all_files_in_folder(analyzer, folder_path, num_cores=None, file_extension=".stl"):
    files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
    files = [os.path.join(folder_path, f) for f in files]
    logging.info(f"Analyzing {len(files)} files in folder: {folder_path}")

    for file in files:
        try:
            results = analyzer.analyze_multiple_files([file], num_cores)
            for result in results:
                if result:
                    for key, value in result.items():
                        print(f"{key}: {value}")
                        
                        # Print PCA results
                        if "Principal Axis" in key:
                            print(f"{key}: {value}")

                    print("\n---\n")
        except Exception as e:
            logging.error(f"An error occurred while analyzing the file {file}: {e}")
            continue

# Create an analyzer object
analyzer = BallastAnalyzer()

# Analyzing all files in a folder
folder_path = "/home/railcmuvpn/Documents/Morphology_CFC/model"
analyze_all_files_in_folder(analyzer, folder_path, num_cores=4)
