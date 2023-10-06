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
import csv
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor
from sklearn.decomposition import PCA

class BallastAnalyzer:
    def __init__(self):
        pass

    def _sphere_fit_function(self, parameters, points):
        center_x, center_y, center_z, radius = parameters
        center = np.array([center_x, center_y, center_z])
        residual = np.linalg.norm(points - center, axis=1) - radius
        return residual

    def _fit_sphere(self, points):
        center = np.mean(points, axis=0)
        radius = np.mean(np.linalg.norm(points - center, axis=1))
        initial_guess = np.array([center[0], center[1], center[2], radius])
        result = least_squares(self._sphere_fit_function, initial_guess, args=(points,))
        return result.x[0:3], result.x[3]

    def _save_point_cloud(self, vertices, filename):
        if not os.path.exists('point_clouds'):
            os.makedirs('point_clouds')
        np.savetxt(f"point_clouds/{filename}_point_cloud.csv", vertices, delimiter=",")

    def _save_mesh(self, mesh, filename):
        if not os.path.exists('meshes'):
            os.makedirs('meshes')
        mesh.export(f"meshes/{filename}.stl")

    def analyze_ballast(self, filename):
        try:
            model = trimesh.load(filename)
            surface_vertices = model.vertices
            center, radius = self._fit_sphere(surface_vertices)
            
            dimensions = model.bounding_box.primitive.extents
            intermediate = np.mean(dimensions)
            shortest = min(dimensions)
            longest = max(dimensions)

            elongation = intermediate / longest
            flatness = shortest / intermediate
            convexity = model.convex_hull.volume / model.volume
            sphericity = (np.pi * (3 * model.volume) / (model.area ** 2)) ** (1/3)
            
            dists = np.linalg.norm(surface_vertices - center, axis=1)
            mean_dist = np.mean(dists - radius)
            roughness = mean_dist
            roundness = mean_dist / radius
            
            aspect_ratio = longest / shortest
            actual_volume = model.volume

            bounding_box = model.bounding_box_oriented
            bounding_box_dimensions = bounding_box.extents
            bounding_box_volume = np.prod(bounding_box_dimensions)

            angularity_index = actual_volume / bounding_box_volume
            
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
                'Volume': actual_volume,  # Added volume
                'Surface Area': model.area  # Added surface area
            }

            # PCA for principal axes of variation
            pca = PCA(n_components=3)
            pca.fit(surface_vertices)
            principal_axes = pca.components_

            data.update({
                'Principal Axis 1': principal_axes[0].tolist(),
                'Principal Axis 2': principal_axes[1].tolist(),
                'Principal Axis 3': principal_axes[2].tolist()
            })

            csv_file_path = 'data.csv'
            if not os.path.isfile(csv_file_path):
                with open(csv_file_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=data.keys())
                    writer.writeheader()

            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=data.keys())
                writer.writerow(data)
            
            self._save_point_cloud(surface_vertices, os.path.basename(filename).split('.')[0])
            self._save_mesh(model, os.path.basename(filename).split('.')[0])

            return data
        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")
            return None

    def analyze_multiple_files(self, filenames, num_cores=None):
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(executor.map(self.analyze_ballast, filenames))
        return results
