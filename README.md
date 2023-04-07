- # morphology_analysis
	- ## this is the morphology analysis
		the CMURail CFC mophology analysis is the tool for analyze the ballast or 3D object. We have developed a Python package that analyzes the morphology of ballast samples. The package uses the `trimesh` library to process 3D models of the samples and calculates various morphological properties, such as elongation, flatness, convexity, sphericity, roughness, and roundness. The package is designed to process multiple 3D models in a folder, saving the results to a CSV file for further analysis.

		Throughout our conversation, we have discussed various aspects of the package, including refactoring the code, improving the analysis, and creating a package structure. We have also covered the addition of grain size distribution, porosity, and permeability factors, which are not included in the current package because they require experimental data from the lab. The package is now more organized and easier to use, with clearer instructions and documentation.

		To make the package accessible and easy to use, make sure to provide a comprehensive README file with instructions on how to set up the environment, install the required packages, and use the package. A requirements.txt file should also be included to facilitate the installation of the necessary dependencies.

		This project provides a valuable tool for analyzing ballast samples and can be further expanded to include more advanced analysis methods or integrate experimental data for a more comprehensive understanding of the samples' properties.

	- ## Scope of our package
		- analyze Intermediate
		- analyze Shortest
		- analyze Longest
		- analyze Elongation
		- analyze Flatness
		- analyze Convexity
	 	- analyze Sphericity
		- analyze Roundness
		- analyze Roughness
		- analyze Sphere center
		- analyze Sphere radius
		- analyze Sphere fit
		- analyze Angularity Index
	 	- analyze Aspect Ratio

	- ## prerequirement
		- python 3
		- numpy
		- scipy
		- trimesh
		- plotly (for visulaize)
	- ## to install the environment (if you don't have environtment)
		- scenario 1 (for who have Anaconda or conda)
			- to install the environment into the conda please use
			-
			  ```
			  conda env create -f environment.yml
			  ```
			- to use the environment
			-
			  ```
			  conda activate cenv
			  ```
		- scenario 2 (for normal python profile)
			- Use `pip` to install the environment in `requirements.txt` follow in this command
			-
			  ```
			  pip install -r requirements.txt
			  ```
	- ## to run the code:
			-
			  ```
			  python3 test.py
			  ```
	- ## NOTE
		- please make sure the the function that you use is collect if you use the code for analyze the `.obj` just avoid this step, but if other please edit the eight line to you collect 3d type file in `test.py`.
		- there are 2 option to run the code in `test.py` if you are run each single file please use `option 2` (that I already use). whereas you need to delete the `'''` in line 7 and 40 and/or `'''` or delete the `option 2` to use `option 1` for analyze a hole folder.
		- in the `option 1` you need to get direction the folder which is the directory for store the 3d file. in line 38 follow in `line 38`.
				  ``` python
				  folder_path = "path/to/your/folder"
				  ```
		- However the `option2` need to edit the direction in here`line 42`:
				  ``` python
				  result = analyze_ballast('F15_1/untitled.obj')
				  ```
		- the result will be in term of the `data.csv` you can edit the direction and name in`ballast_analyzer` from `analyzer.py`
