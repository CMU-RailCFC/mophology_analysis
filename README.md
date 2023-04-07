- # morphology_analysis
	- ## this is the morphology analysis
	- prerequirement
		- python version 3
		- numpy
		- scipy
		- trimesh
		- plotly (for visulaize)
	- to install the environment
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
		- to run the code:
			-
			  ```
			  python3 test.py
			  ```
		- ### NOTE
			- please make sure the the function that you use is collect if you use the code for analyze the `.obj` just avoid this step, but if other please edit the eight line to you collect 3d type file in `test.py`.
			- there are 2 option to run the code in `test.py` if you are run each single file please use `option 2` (that I already use). whereas you need to delete the `'''` in line 7 and 40 and/or `'''` or delete the `option 2`.
				- in the `option 1` you need to get direction the folder which is the directory for store the 3d file. in line 38 follow in `line 38`.
				-
				  ``` python
				  folder_path = "path/to/your/folder"
				  ```
				- However the `option2` need to edit the direction in here`line 42`:
				-
				  ``` python
				  result = analyze_ballast('F15_1/untitled.obj')
				  ```
			- the result will be in term of the `data.csv` you can edit the direction and name in`ballast_analyzer` from `analyzer.py`
