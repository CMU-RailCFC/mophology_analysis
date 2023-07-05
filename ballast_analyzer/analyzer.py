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
'''


import os
import trimesh
import numpy as np
from scipy.optimize import least_squares
import csv


def sphere_fit_function(parameters, points):
    center_x, center_y, center_z, radius = parameters
    center = np.array([center_x, center_y, center_z])
    residual = np.linalg.norm(points - center, axis=1) - radius
    return residual


def fit_sphere(points):
    center = np.mean(points, axis=0)
    radius = np.mean(np.linalg.norm(points, axis=1))
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
    surface_area = model.area

    elongation = longest / intermediate
    flatness = (shortest / intermediate) ** 2
    convexity = model.convex_hull.volume / model.volume
    sphericity = (3 * (4 * np.pi * model.volume) / (surface_area ** 2)) ** (1/3)

    dists = np.linalg.norm(surface_vertices - center, axis=1)
    mean_dist = np.mean(dists - radius)

    surface_vertices = model.vertices
    convex_hull = model.convex_hull
    dists = np.linalg.norm(surface_vertices - center, axis=1)
    mean_dist = np.mean(dists - radius)

    roughness = mean_dist
    roundness = 1 - (mean_dist / radius)

    aspect_ratio = longest / shortest

    # Calculate the volume of the bounding box
    bounding_box_volume = dimensions[0] * dimensions[1] * dimensions[2]

    # Calculate the Angularity Index (AI)
    angularity_index = model.volume / bounding_box_volume

    data = {'Filename': filename,
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


if __name__ == "__main__":
    result = analyze_ballast('F15_1/untitled.obj')
    print(result)
