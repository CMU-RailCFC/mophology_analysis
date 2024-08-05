'''
Before you using the file please go to cite some of the resources that I have use in the sourcecode

to calculate the roundness of a 3D model, you might use a method such as the one described in "A roundness estimation method for 3D digital models" by C. H. Wu and H. J. Chung. This method involves fitting a sphere to the model and calculating the average distance between the sphere and the model's surface.

Alternatively, you might use a method such as the one described in "Roughness measurement of 3D models" by Y. T. Chen et al., which involves calculating the mean square slope of the model's surface.

Trimesh documentation. "Trimesh 2.38.14." PyMesh, http://trimsh.org/.

Scipy documentation. "least_squares." Scipy, https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html.

However, this quite be abitdifference from both text book there are some function that have to optimize the source code and there are some of part that implement with myself

THIS SOURCECODE IS IMPLEMENT BY S.THEERADON PRODUCE BY CMURAIL CFC TEAM

THE KNOWLEDGE OF SOURCE WILL BE ONLY CMURAIL CFC

# please read on readme before you run source code
## please check the evironment and packet is cover in the requirement.txt (manualto install is in readme)

NO COMERCAIL LICENCE ANYMORE


for cite or reference: 10.1088/1755-1315/1332/1/012016
'''


import os
import trimesh
import numpy as np
import csv
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor

np.random.seed(0)  # Set random seed for reproducibility

def sphere_fit_function(parameters, points):
    center_x, center_y, center_z, radius = parameters
    center = np.array([center_x, center_y, center_z])
    residual = np.linalg.norm(points - center, axis=1) - radius
    return residual

def fit_sphere(points):
    center = np.mean(points, axis=0)
    radius = np.mean(np.linalg.norm(points - center, axis=1))
    initial_guess = np.array([center[0], center[1], center[2], radius])
    result = least_squares(sphere_fit_function, initial_guess, args=(points,))
    return result.x[0:3], result.x[3]

def analyze_ballast(filename):
    model = trimesh.load(filename)
    surface_vertices = model.vertices
    center, radius = fit_sphere(surface_vertices)
    
    dimensions = model.bounding_box.primitive.extents
    intermediate = (dimensions[0] + dimensions[1] + dimensions[2]) / 3
    shortest = min(dimensions)
    longest = max(dimensions)
    
    elongation = intermediate / longest
    flatness = shortest / intermediate
    convexity = model.convex_hull.volume / model.volume
    sphericity = (3 * (4 * np.pi * model.volume) / (model.area ** 2)) ** (1/3)
    
    dists = np.linalg.norm(surface_vertices - center, axis=1)
    mean_dist = np.mean(dists - radius)
    roughness = mean_dist
    roundness = (mean_dist / radius)
    
    aspect_ratio = longest / shortest
    actual_volume = model.volume
    
    bounding_box = model.bounding_box_oriented
    bounding_box_dimensions = bounding_box.extents
    bounding_box_volume = np.prod(bounding_box_dimensions)
    
    angularity_index = actual_volume / bounding_box_volume
    
    # Adding bounding box orientation analysis
    bounding_box_orientation = bounding_box.primitive.transform
    orientation_x = bounding_box_orientation[0, 0]
    orientation_y = bounding_box_orientation[1, 1]
    orientation_z = bounding_box_orientation[2, 2]
    
    data = {
        'Filename': filename,
        'Intermediate': intermediate,
        'Shortest': shortest,
        'Longest': longest,
        'Elongation': elongation,
        'Flatness': flatness,
        'Convexity': convexity,
        'Sphericity': sphericity,
        'Roundness': roundness,
        'Roughness': roughness,
        'Center X': center[0],
        'Center Y': center[1],
        'Center Z': center[2],
        'Radius': radius,
        'Aspect Ratio': aspect_ratio,
        'Angularity Index': angularity_index,
        'Orientation X': orientation_x,
        'Orientation Y': orientation_y,
        'Orientation Z': orientation_z,
    }
    
    csv_file_path = 'data.csv'
    if not os.path.isfile(csv_file_path):
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data.keys())
            writer.writeheader()

    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writerow(data)
    
    return data

def analyze_multiple_files(filenames, num_cores=None):
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(analyze_ballast, filenames))
    return results
