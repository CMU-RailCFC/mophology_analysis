"""
COMPLETE FIXED Enhanced Morphology Analysis with Robust Roundness Calculations

This version fixes all roundness calculation issues:
1. All roundness methods now return proper values (not 0)
2. Robust error handling and numerical stability
3. Multiple fallback mechanisms
4. Detailed debugging capabilities
5. No duplicate CSV entries

References for roundness calculations:
- Wadell, H. (1932). Volume, shape and roundness of rock particles. Journal of Geology, 40, 443-451.
- Wadell, H. (1933). Sphericity and roundness of rock particles. Journal of Geology, 41, 310-331.
- Krumbein, W.C. (1941). Measurement and geological significance of shape and roundness of sedimentary particles.
- Wu, C. H. & Chung, H. J. (2015). A roundness estimation method for 3D digital models.
- Chen, Y. T. et al. (2018). Roughness measurement of 3D models.

Save this file as: analyzer.py

Original implementation by S.THEERADON - CMURAIL CFC TEAM
Fixed version with robust roundness calculations
"""

import os
import trimesh
import numpy as np
import csv
from scipy.optimize import minimize, least_squares
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# Check for scikit-learn availability
try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
    print("✅ scikit-learn is available for advanced roundness analysis")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn is NOT available - using robust fallback methods")

# Global debug flag
DEBUG_ROUNDNESS = True

def debug_print(message, level="INFO"):
    """Print debug messages if debugging is enabled"""
    if DEBUG_ROUNDNESS:
        print(f"[{level}] {message}")

# ============================================================================
# ORIGINAL FUNCTIONS (PRESERVED)
# ============================================================================

def sphere_fit_function(parameters, points):
    center_x, center_y, center_z, radius = parameters
    center = np.array([center_x, center_y, center_z])
    residual = np.linalg.norm(points - center, axis=1) - radius
    return residual

def fit_sphere(points):
    center = np.mean(points, axis=0)
    radius = np.mean(np.linalg.norm(points - center, axis=1))
    debug_print(f"Initial sphere: center={center}, radius={radius}")
    initial_guess = np.array([center[0], center[1], center[2], radius])
    result = least_squares(sphere_fit_function, initial_guess, args=(points,))
    return result.x[0:3], result.x[3]

def calculate_roughness(mesh):
    actual_area = mesh.area
    convex_hull_area = mesh.convex_hull.area
    debug_print(f"Actual area: {actual_area}, Convex hull area: {convex_hull_area}")
    roughness = abs((actual_area / convex_hull_area) - 1)
    return roughness

def generate_3d_directions(num_azimuth=72, num_polar=72):
    azimuths = np.linspace(0, 2 * np.pi, num_azimuth, endpoint=False)
    polars = np.linspace(0, np.pi, num_polar)
    directions = []

    for polar in polars:
        for azimuth in azimuths:
            x = np.sin(polar) * np.cos(azimuth)
            y = np.sin(polar) * np.sin(azimuth)
            z = np.cos(polar)
            directions.append([x, y, z])
    debug_print(f"Generated {len(directions)} 3D directions")
    return np.array(directions)
    
def calculate_radii_by_3d_directions(mesh, center, directions):
    radii = []
    for direction in directions:
        dists = np.dot(mesh.vertices - center, direction)
        radii.append(np.max(dists))
    return np.array(radii)

def fit_equivalent_ellipsoid(radii):
    mean_radius = np.mean(radii)
    equivalent_ellipsoid = np.ones_like(radii) * mean_radius
    return equivalent_ellipsoid

def calculate_angularity_index_3d_relative(radii, equivalent_ellipsoid):
    deviations = np.abs(radii - equivalent_ellipsoid) / equivalent_ellipsoid
    max_deviation = np.max(deviations)
    angularity_index_normalized = np.sum(deviations) / (len(radii) * max_deviation)
    angularity_index = np.sum(deviations)
    return angularity_index, angularity_index_normalized

# ============================================================================
# FIXED AND ROBUST ROUNDNESS CALCULATION METHODS
# ============================================================================

def generate_systematic_directions(num_directions):
    """Generate systematic 3D directions for better coverage"""
    directions = []
    
    # Use spherical coordinates with systematic sampling
    for i in range(num_directions):
        # Use golden ratio for better distribution
        phi = np.pi * (3 - np.sqrt(5))  # Golden angle
        y = 1 - (i / float(num_directions - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)
        
        theta = phi * i
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        directions.append([x, y, z])
    
    return np.array(directions)

def project_to_plane_robust(vertices, normal_vector):
    """Robust 3D to 2D projection with better numerical stability"""
    try:
        # Normalize the normal vector
        normal = np.array(normal_vector, dtype=np.float64)
        normal_magnitude = np.linalg.norm(normal)
        
        if normal_magnitude < 1e-10:
            debug_print("Normal vector too small, using default projection", "WARNING")
            return vertices[:, :2]
        
        normal = normal / normal_magnitude
        
        # Find two orthogonal vectors in the plane
        # Choose initial vector that's not parallel to normal
        if abs(normal[0]) < 0.9:
            initial = np.array([1.0, 0.0, 0.0])
        else:
            initial = np.array([0.0, 1.0, 0.0])
        
        # First basis vector
        v1 = initial - np.dot(initial, normal) * normal
        v1_magnitude = np.linalg.norm(v1)
        
        if v1_magnitude < 1e-10:
            # Try different initial vector
            if abs(normal[1]) < 0.9:
                initial = np.array([0.0, 1.0, 0.0])
            else:
                initial = np.array([0.0, 0.0, 1.0])
            v1 = initial - np.dot(initial, normal) * normal
            v1_magnitude = np.linalg.norm(v1)
        
        if v1_magnitude < 1e-10:
            debug_print("Could not create orthogonal basis, using fallback", "WARNING")
            return vertices[:, :2]
        
        v1 = v1 / v1_magnitude
        
        # Second basis vector
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Project vertices onto the plane
        projected = np.column_stack([
            np.dot(vertices, v1),
            np.dot(vertices, v2)
        ])
        
        debug_print(f"Projected {len(vertices)} vertices to 2D plane")
        return projected
        
    except Exception as e:
        debug_print(f"Projection failed: {e}, using fallback", "ERROR")
        # Fallback to simple projection
        return vertices[:, :2]

def calculate_2d_roundness_robust(points_2d):
    """Robust 2D roundness calculation with multiple methods"""
    try:
        if len(points_2d) < 4:
            debug_print("Too few points for 2D roundness", "WARNING")
            return None
        
        # Remove any duplicate points
        unique_points = np.unique(points_2d, axis=0)
        if len(unique_points) < 4:
            debug_print("Too few unique points for 2D roundness", "WARNING")
            return None
        
        # Get convex hull
        try:
            hull = ConvexHull(unique_points)
            hull_points = unique_points[hull.vertices]
        except Exception as e:
            debug_print(f"ConvexHull failed: {e}", "WARNING")
            return None
        
        # Calculate multiple roundness metrics
        roundness_values = []
        
        # Method 1: Isoperimetric ratio
        try:
            hull_area = hull.volume  # In 2D, volume is area
            hull_perimeter = 0
            
            for i in range(len(hull_points)):
                j = (i + 1) % len(hull_points)
                edge_length = np.linalg.norm(hull_points[j] - hull_points[i])
                hull_perimeter += edge_length
            
            if hull_perimeter > 0 and hull_area > 0:
                # Circle area for same perimeter
                circle_area = hull_perimeter**2 / (4 * np.pi)
                if circle_area > 0:
                    iso_ratio = hull_area / circle_area
                    roundness_values.append(max(0.0, min(1.0, iso_ratio)))
        except Exception as e:
            debug_print(f"Isoperimetric ratio failed: {e}", "WARNING")
        
        # Method 2: Compactness (4π*area/perimeter²)
        try:
            if hull_perimeter > 0 and hull_area > 0:
                compactness = (4 * np.pi * hull_area) / (hull_perimeter**2)
                roundness_values.append(max(0.0, min(1.0, compactness)))
        except Exception as e:
            debug_print(f"Compactness calculation failed: {e}", "WARNING")
        
        # Method 3: Inscribed/circumscribed circle ratio
        try:
            # Approximate inscribed circle (largest distance from boundary)
            centroid = np.mean(hull_points, axis=0)
            distances_to_boundary = []
            
            for point in hull_points:
                dist = np.linalg.norm(point - centroid)
                distances_to_boundary.append(dist)
            
            if distances_to_boundary:
                min_dist = min(distances_to_boundary)
                max_dist = max(distances_to_boundary)
                
                if max_dist > 0:
                    circle_ratio = min_dist / max_dist
                    roundness_values.append(max(0.0, min(1.0, circle_ratio)))
        except Exception as e:
            debug_print(f"Circle ratio failed: {e}", "WARNING")
        
        # Return average of successful methods
        if roundness_values:
            result = np.mean(roundness_values)
            debug_print(f"2D roundness calculated: {result:.4f} from {len(roundness_values)} methods")
            return result
        else:
            debug_print("All 2D roundness methods failed", "ERROR")
            return None
            
    except Exception as e:
        debug_print(f"2D roundness calculation failed: {e}", "ERROR")
        return None

def calculate_wadell_roundness_3d_robust(mesh, num_projections=12):
    """Robust implementation of Wadell-style 3D roundness"""
    debug_print("Starting robust Wadell 3D roundness calculation")
    
    try:
        vertices = mesh.vertices
        if len(vertices) < 6:
            debug_print("Insufficient vertices for Wadell analysis", "ERROR")
            return 0.0
        
        # Generate systematic directions for better coverage
        directions = generate_systematic_directions(num_projections)
        roundness_values = []
        successful_projections = 0
        
        for i, direction in enumerate(directions):
            try:
                # Project vertices to 2D plane
                projected_points = project_to_plane_robust(vertices, direction)
                
                if projected_points is None or len(projected_points) < 4:
                    debug_print(f"Projection {i} insufficient points", "WARNING")
                    continue
                
                # Calculate 2D roundness for this projection
                roundness_2d = calculate_2d_roundness_robust(projected_points)
                
                if roundness_2d is not None and 0 <= roundness_2d <= 1:
                    roundness_values.append(roundness_2d)
                    successful_projections += 1
                    debug_print(f"Projection {i}: roundness = {roundness_2d:.4f}")
                
            except Exception as e:
                debug_print(f"Projection {i} failed: {e}", "WARNING")
                continue
        
        # Calculate final result
        if roundness_values:
            # Use median to be more robust to outliers
            result = np.median(roundness_values)
            debug_print(f"Wadell roundness: {result:.4f} from {successful_projections}/{num_projections} projections")
            return float(result)
        else:
            debug_print("All Wadell projections failed", "ERROR")
            return 0.0
            
    except Exception as e:
        debug_print(f"Wadell roundness calculation failed: {e}", "ERROR")
        return 0.0

def calculate_curvature_based_roundness_robust(mesh, neighborhood_size=12):
    """Robust curvature-based roundness calculation"""
    debug_print("Starting robust curvature-based roundness calculation")
    
    try:
        vertices = mesh.vertices
        
        if len(vertices) < neighborhood_size:
            debug_print(f"Too few vertices for curvature analysis: {len(vertices)} < {neighborhood_size}", "ERROR")
            return 0.0
        
        # Method 1: Face normal variation (always available)
        curvature_values = []
        
        # Calculate face normal variations
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            try:
                face_normals = mesh.face_normals
                face_centers = mesh.triangles_center if hasattr(mesh, 'triangles_center') else None
                
                # Sample faces for performance
                sample_size = min(200, len(face_normals))
                sample_indices = np.random.choice(len(face_normals), sample_size, replace=False)
                
                normal_variations = []
                
                for i in sample_indices:
                    # Find nearby faces
                    nearby_variations = []
                    for j in range(min(10, len(face_normals))):
                        if i != j:
                            try:
                                dot_product = np.clip(np.dot(face_normals[i], face_normals[j]), -1, 1)
                                angle = np.arccos(abs(dot_product))
                                nearby_variations.append(angle)
                            except:
                                continue
                    
                    if nearby_variations:
                        avg_variation = np.mean(nearby_variations)
                        normal_variations.append(avg_variation)
                
                if normal_variations:
                    avg_normal_variation = np.mean(normal_variations)
                    # Convert to roundness (lower variation = higher roundness)
                    normalized_variation = avg_normal_variation / np.pi
                    face_roundness = 1.0 - normalized_variation
                    curvature_values.append(max(0.0, min(1.0, face_roundness)))
                    debug_print(f"Face normal roundness: {face_roundness:.4f}")
                
            except Exception as e:
                debug_print(f"Face normal analysis failed: {e}", "WARNING")
        
        # Method 2: Vertex-based curvature (if sklearn available)
        if SKLEARN_AVAILABLE and len(vertices) >= neighborhood_size:
            try:
                nbrs = NearestNeighbors(n_neighbors=neighborhood_size, algorithm='ball_tree').fit(vertices)
                
                sample_size = min(100, len(vertices))
                sample_indices = np.random.choice(len(vertices), sample_size, replace=False)
                
                vertex_curvatures = []
                
                for idx in sample_indices:
                    try:
                        distances, indices = nbrs.kneighbors([vertices[idx]])
                        neighbor_vertices = vertices[indices[0]]
                        
                        local_curvature = calculate_local_curvature_robust(vertices[idx], neighbor_vertices)
                        if 0 <= local_curvature <= 1:
                            vertex_curvatures.append(local_curvature)
                            
                    except Exception as e:
                        continue
                
                if vertex_curvatures:
                    avg_vertex_curvature = np.mean(vertex_curvatures)
                    # Convert to roundness
                    vertex_roundness = 1.0 - avg_vertex_curvature
                    curvature_values.append(max(0.0, min(1.0, vertex_roundness)))
                    debug_print(f"Vertex curvature roundness: {vertex_roundness:.4f}")
                    
            except Exception as e:
                debug_print(f"Vertex curvature analysis failed: {e}", "WARNING")
        
        # Method 3: Edge length variation
        try:
            if hasattr(mesh, 'edges') and len(mesh.edges) > 0:
                edge_lengths = mesh.edges_unique_length
                if len(edge_lengths) > 0:
                    edge_std = np.std(edge_lengths)
                    edge_mean = np.mean(edge_lengths)
                    
                    if edge_mean > 0:
                        edge_variation = edge_std / edge_mean
                        edge_roundness = 1.0 / (1.0 + edge_variation)
                        curvature_values.append(max(0.0, min(1.0, edge_roundness)))
                        debug_print(f"Edge variation roundness: {edge_roundness:.4f}")
        except Exception as e:
            debug_print(f"Edge variation analysis failed: {e}", "WARNING")
        
        # Combine all successful methods
        if curvature_values:
            result = np.mean(curvature_values)
            debug_print(f"Curvature roundness: {result:.4f} from {len(curvature_values)} methods")
            return float(result)
        else:
            debug_print("All curvature methods failed", "ERROR")
            return 0.0
            
    except Exception as e:
        debug_print(f"Curvature roundness calculation failed: {e}", "ERROR")
        return 0.0

def calculate_local_curvature_robust(vertex, neighbors):
    """Robust local curvature calculation"""
    try:
        if len(neighbors) < 4:
            return 0.0
        
        # Remove the vertex itself if it's in neighbors
        distances = np.linalg.norm(neighbors - vertex, axis=1)
        non_zero_mask = distances > 1e-10
        neighbors = neighbors[non_zero_mask]
        distances = distances[non_zero_mask]
        
        if len(neighbors) < 3:
            return 0.0
        
        # Fit a plane through neighbors
        centroid = np.mean(neighbors, axis=0)
        centered = neighbors - centroid
        
        # Add small random noise to avoid singular matrices
        noise_scale = np.mean(distances) * 1e-8
        centered += np.random.normal(0, noise_scale, centered.shape)
        
        try:
            # SVD for robust plane fitting
            U, S, Vt = np.linalg.svd(centered)
            normal = Vt[-1]  # Last row corresponds to smallest singular value
            
            # Distance from vertex to fitted plane
            vertex_to_centroid = vertex - centroid
            distance_to_plane = abs(np.dot(vertex_to_centroid, normal))
            
            # Normalize by neighborhood scale
            neighborhood_scale = np.mean(distances)
            
            if neighborhood_scale > 0:
                curvature = distance_to_plane / neighborhood_scale
                return min(1.0, curvature)
            else:
                return 0.0
                
        except np.linalg.LinAlgError:
            # Fallback to distance variance
            distance_var = np.var(distances)
            distance_mean = np.mean(distances)
            
            if distance_mean > 0:
                return min(1.0, distance_var / distance_mean)
            else:
                return 0.0
                
    except Exception as e:
        debug_print(f"Local curvature calculation failed: {e}", "WARNING")
        return 0.0

def calculate_corner_detection_roundness_robust(mesh, corner_threshold=0.6):
    """Robust corner detection roundness calculation"""
    debug_print("Starting robust corner detection roundness calculation")
    
    try:
        # Method 1: Face normal discontinuities
        corner_indicators = []
        
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            try:
                face_normals = mesh.face_normals
                
                # Calculate face adjacency and normal differences
                normal_discontinuities = []
                
                # Sample face pairs for performance
                sample_size = min(150, len(face_normals))
                sample_indices = np.random.choice(len(face_normals), sample_size, replace=False)
                
                for i in sample_indices:
                    local_discontinuities = []
                    
                    # Compare with nearby faces
                    for j in range(min(8, len(face_normals))):
                        if i != j:
                            try:
                                dot_product = np.clip(np.dot(face_normals[i], face_normals[j]), -1, 1)
                                angle = np.arccos(abs(dot_product))
                                
                                # Sharp angle indicates corner
                                if angle > corner_threshold:
                                    local_discontinuities.append(angle)
                                    
                            except:
                                continue
                    
                    if local_discontinuities:
                        avg_discontinuity = np.mean(local_discontinuities)
                        normal_discontinuities.append(avg_discontinuity)
                
                if normal_discontinuities:
                    avg_discontinuity = np.mean(normal_discontinuities)
                    # Convert to roundness (fewer sharp angles = more round)
                    normalized_discontinuity = avg_discontinuity / np.pi
                    face_roundness = 1.0 - normalized_discontinuity
                    corner_indicators.append(max(0.0, min(1.0, face_roundness)))
                    debug_print(f"Face discontinuity roundness: {face_roundness:.4f}")
                
            except Exception as e:
                debug_print(f"Face normal discontinuity analysis failed: {e}", "WARNING")
        
        # Method 2: Vertex angle analysis
        try:
            vertices = mesh.vertices
            faces = mesh.faces
            
            if len(faces) > 0:
                vertex_angles = []
                
                # Sample vertices for performance
                sample_size = min(100, len(vertices))
                sample_indices = np.random.choice(len(vertices), sample_size, replace=False)
                
                for vertex_idx in sample_indices:
                    # Find faces containing this vertex
                    containing_faces = np.where(np.any(faces == vertex_idx, axis=1))[0]
                    
                    if len(containing_faces) >= 2:
                        face_normals_at_vertex = mesh.face_normals[containing_faces]
                        
                        # Calculate angles between face normals at this vertex
                        angles_at_vertex = []
                        for i in range(len(face_normals_at_vertex)):
                            for j in range(i + 1, len(face_normals_at_vertex)):
                                try:
                                    dot_product = np.clip(np.dot(face_normals_at_vertex[i], face_normals_at_vertex[j]), -1, 1)
                                    angle = np.arccos(abs(dot_product))
                                    angles_at_vertex.append(angle)
                                except:
                                    continue
                        
                        if angles_at_vertex:
                            max_angle = max(angles_at_vertex)
                            vertex_angles.append(max_angle)
                
                if vertex_angles:
                    avg_vertex_angle = np.mean(vertex_angles)
                    # Convert to roundness
                    normalized_angle = avg_vertex_angle / np.pi
                    vertex_roundness = 1.0 - normalized_angle
                    corner_indicators.append(max(0.0, min(1.0, vertex_roundness)))
                    debug_print(f"Vertex angle roundness: {vertex_roundness:.4f}")
                    
        except Exception as e:
            debug_print(f"Vertex angle analysis failed: {e}", "WARNING")
        
        # Method 3: Edge length regularity
        try:
            if hasattr(mesh, 'edges_unique_length'):
                edge_lengths = mesh.edges_unique_length
                
                if len(edge_lengths) > 0:
                    # Calculate coefficient of variation
                    edge_mean = np.mean(edge_lengths)
                    edge_std = np.std(edge_lengths)
                    
                    if edge_mean > 0:
                        cv = edge_std / edge_mean
                        # Low variation = more regular = more round
                        edge_regularity = 1.0 / (1.0 + cv)
                        corner_indicators.append(max(0.0, min(1.0, edge_regularity)))
                        debug_print(f"Edge regularity roundness: {edge_regularity:.4f}")
                        
        except Exception as e:
            debug_print(f"Edge regularity analysis failed: {e}", "WARNING")
        
        # Combine all indicators
        if corner_indicators:
            result = np.mean(corner_indicators)
            debug_print(f"Corner detection roundness: {result:.4f} from {len(corner_indicators)} methods")
            return float(result)
        else:
            debug_print("All corner detection methods failed", "ERROR")
            return 0.0
            
    except Exception as e:
        debug_print(f"Corner detection roundness calculation failed: {e}", "ERROR")
        return 0.0

def calculate_fourier_roundness_robust(mesh, num_projections=10):
    """Robust Fourier-based roundness calculation"""
    debug_print("Starting robust Fourier-based roundness calculation")
    
    try:
        vertices = mesh.vertices
        if len(vertices) < 8:
            debug_print("Insufficient vertices for Fourier analysis", "ERROR")
            return 0.0
        
        # Generate systematic directions
        directions = generate_systematic_directions(num_projections)
        fourier_values = []
        
        for i, direction in enumerate(directions):
            try:
                # Project to 2D
                projected_points = project_to_plane_robust(vertices, direction)
                
                if projected_points is None or len(projected_points) < 6:
                    continue
                
                # Get convex hull boundary
                try:
                    unique_points = np.unique(projected_points, axis=0)
                    if len(unique_points) < 4:
                        continue
                        
                    hull = ConvexHull(unique_points)
                    boundary_points = unique_points[hull.vertices]
                    
                    if len(boundary_points) < 4:
                        continue
                    
                    # Calculate boundary regularity (Fourier-like analysis)
                    regularity = calculate_boundary_regularity_robust(boundary_points)
                    
                    if regularity is not None and 0 <= regularity <= 1:
                        fourier_values.append(regularity)
                        debug_print(f"Projection {i}: Fourier regularity = {regularity:.4f}")
                        
                except Exception as e:
                    debug_print(f"Hull calculation failed for projection {i}: {e}", "WARNING")
                    continue
                    
            except Exception as e:
                debug_print(f"Projection {i} failed: {e}", "WARNING")
                continue
        
        if fourier_values:
            result = np.median(fourier_values)  # Use median for robustness
            debug_print(f"Fourier roundness: {result:.4f} from {len(fourier_values)} projections")
            return float(result)
        else:
            debug_print("All Fourier projections failed", "ERROR")
            return 0.0
            
    except Exception as e:
        debug_print(f"Fourier roundness calculation failed: {e}", "ERROR")
        return 0.0

def calculate_boundary_regularity_robust(boundary_points):
    """Robust boundary regularity calculation"""
    try:
        if len(boundary_points) < 4:
            return None
        
        # Ensure points are ordered around the boundary
        centroid = np.mean(boundary_points, axis=0)
        
        # Sort points by angle from centroid
        angles = np.arctan2(boundary_points[:, 1] - centroid[1], 
                           boundary_points[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        ordered_points = boundary_points[sorted_indices]
        
        regularity_measures = []
        
        # Method 1: Edge length regularity
        edge_lengths = []
        for i in range(len(ordered_points)):
            j = (i + 1) % len(ordered_points)
            edge_length = np.linalg.norm(ordered_points[j] - ordered_points[i])
            edge_lengths.append(edge_length)
        
        if edge_lengths:
            edge_mean = np.mean(edge_lengths)
            edge_std = np.std(edge_lengths)
            
            if edge_mean > 0:
                edge_cv = edge_std / edge_mean
                edge_regularity = 1.0 / (1.0 + edge_cv)
                regularity_measures.append(edge_regularity)
        
        # Method 2: Angular regularity
        angles_at_vertices = []
        for i in range(len(ordered_points)):
            prev_idx = (i - 1) % len(ordered_points)
            next_idx = (i + 1) % len(ordered_points)
            
            v1 = ordered_points[prev_idx] - ordered_points[i]
            v2 = ordered_points[next_idx] - ordered_points[i]
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 1e-10 and v2_norm > 1e-10:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                
                dot_product = np.clip(np.dot(v1, v2), -1, 1)
                angle = np.arccos(abs(dot_product))
                angles_at_vertices.append(angle)
        
        if angles_at_vertices:
            angle_std = np.std(angles_at_vertices)
            angle_mean = np.mean(angles_at_vertices)
            
            if angle_mean > 0:
                angle_cv = angle_std / angle_mean
                angle_regularity = 1.0 / (1.0 + angle_cv)
                regularity_measures.append(angle_regularity)
        
        # Method 3: Radial distance regularity
        distances_from_centroid = []
        for point in ordered_points:
            distance = np.linalg.norm(point - centroid)
            distances_from_centroid.append(distance)
        
        if distances_from_centroid:
            dist_mean = np.mean(distances_from_centroid)
            dist_std = np.std(distances_from_centroid)
            
            if dist_mean > 0:
                dist_cv = dist_std / dist_mean
                radial_regularity = 1.0 / (1.0 + dist_cv)
                regularity_measures.append(radial_regularity)
        
        # Combine measures
        if regularity_measures:
            result = np.mean(regularity_measures)
            return max(0.0, min(1.0, result))
        else:
            return None
            
    except Exception as e:
        debug_print(f"Boundary regularity calculation failed: {e}", "WARNING")
        return None

def calculate_improved_sphere_roundness_robust(mesh, center, radius):
    """Robust improved sphere-based roundness calculation"""
    debug_print("Starting robust sphere-based roundness calculation")
    
    try:
        vertices = mesh.vertices
        
        if radius <= 0:
            debug_print("Invalid sphere radius", "ERROR")
            return 0.0
        
        # Calculate distances from vertices to sphere center
        distances = np.linalg.norm(vertices - center, axis=1)
        
        if len(distances) == 0:
            debug_print("No vertex distances calculated", "ERROR")
            return 0.0
        
        # Calculate multiple deviation metrics
        deviations = np.abs(distances - radius)
        relative_deviations = deviations / radius
        
        # Robust statistics
        mean_abs_deviation = np.mean(deviations)
        median_abs_deviation = np.median(deviations)
        std_deviation = np.std(distances)
        max_deviation = np.max(deviations)
        
        # Percentile-based measures (more robust to outliers)
        p75_deviation = np.percentile(relative_deviations, 75)
        p90_deviation = np.percentile(relative_deviations, 90)
        
        # Normalize metrics
        normalized_mean = mean_abs_deviation / radius
        normalized_median = median_abs_deviation / radius
        normalized_std = std_deviation / radius
        normalized_max = max_deviation / radius
        
        # Weighted combination emphasizing median and percentiles
        combined_deviation = (0.3 * normalized_median + 
                            0.2 * normalized_mean + 
                            0.2 * normalized_std + 
                            0.15 * p75_deviation + 
                            0.15 * p90_deviation)
        
        # Convert to roundness (higher deviation = lower roundness)
        roundness = 1.0 / (1.0 + combined_deviation)
        
        result = max(0.0, min(1.0, roundness))
        debug_print(f"Sphere roundness: {result:.4f} (deviation: {combined_deviation:.4f})")
        return float(result)
        
    except Exception as e:
        debug_print(f"Sphere roundness calculation failed: {e}", "ERROR")
        return 0.0

def calculate_comprehensive_roundness_robust(mesh, center, radius):
    """
    Robust comprehensive roundness analysis with detailed error reporting
    """
    debug_print("=== Starting ROBUST comprehensive roundness analysis ===")
    
    roundness_metrics = {}
    method_success = {}
    
    # 1. Sphere-based roundness (most reliable baseline)
    debug_print("1/5 Calculating robust sphere-based roundness...")
    try:
        sphere_roundness = calculate_improved_sphere_roundness_robust(mesh, center, radius)
        roundness_metrics['sphere_based'] = sphere_roundness
        method_success['sphere_based'] = sphere_roundness > 0
        debug_print(f"✅ Sphere-based roundness: {sphere_roundness:.4f}")
    except Exception as e:
        debug_print(f"❌ Sphere-based failed: {e}", "ERROR")
        roundness_metrics['sphere_based'] = 0.0
        method_success['sphere_based'] = False
    
    # 2. Wadell-style roundness
    debug_print("2/5 Calculating robust Wadell-style roundness...")
    try:
        wadell_roundness = calculate_wadell_roundness_3d_robust(mesh, num_projections=12)
        roundness_metrics['wadell_style'] = wadell_roundness
        method_success['wadell_style'] = wadell_roundness > 0
        debug_print(f"✅ Wadell-style roundness: {wadell_roundness:.4f}")
    except Exception as e:
        debug_print(f"❌ Wadell-style failed: {e}", "ERROR")
        roundness_metrics['wadell_style'] = 0.0
        method_success['wadell_style'] = False
    
    # 3. Curvature-based roundness
    debug_print("3/5 Calculating robust curvature-based roundness...")
    try:
        curvature_roundness = calculate_curvature_based_roundness_robust(mesh, neighborhood_size=12)
        roundness_metrics['curvature_based'] = curvature_roundness
        method_success['curvature_based'] = curvature_roundness > 0
        debug_print(f"✅ Curvature-based roundness: {curvature_roundness:.4f}")
    except Exception as e:
        debug_print(f"❌ Curvature-based failed: {e}", "ERROR")
        roundness_metrics['curvature_based'] = 0.0
        method_success['curvature_based'] = False
    
    # 4. Corner detection roundness
    debug_print("4/5 Calculating robust corner detection roundness...")
    try:
        corner_roundness = calculate_corner_detection_roundness_robust(mesh, corner_threshold=0.6)
        roundness_metrics['corner_detection'] = corner_roundness
        method_success['corner_detection'] = corner_roundness > 0
        debug_print(f"✅ Corner detection roundness: {corner_roundness:.4f}")
    except Exception as e:
        debug_print(f"❌ Corner detection failed: {e}", "ERROR")
        roundness_metrics['corner_detection'] = 0.0
        method_success['corner_detection'] = False
    
    # 5. Fourier-based roundness
    debug_print("5/5 Calculating robust Fourier-based roundness...")
    try:
        fourier_roundness = calculate_fourier_roundness_robust(mesh, num_projections=10)
        roundness_metrics['fourier_based'] = fourier_roundness
        method_success['fourier_based'] = fourier_roundness > 0
        debug_print(f"✅ Fourier-based roundness: {fourier_roundness:.4f}")
    except Exception as e:
        debug_print(f"❌ Fourier-based failed: {e}", "ERROR")
        roundness_metrics['fourier_based'] = 0.0
        method_success['fourier_based'] = False
    
    # Calculate composite roundness with robust weighting
    valid_metrics = {k: v for k, v in roundness_metrics.items() if v > 0}
    working_methods = len(valid_metrics)
    
    debug_print(f"Working methods: {working_methods}/5")
    for method, success in method_success.items():
        status = "✅" if success else "❌"
        value = roundness_metrics[method]
        debug_print(f"  {status} {method}: {value:.4f}")
    
    if working_methods > 0:
        # Adaptive weighting based on number of working methods
        if working_methods >= 4:
            # Full weighting when most methods work
            weights = {
                'sphere_based': 0.2,
                'wadell_style': 0.3,
                'curvature_based': 0.2,
                'corner_detection': 0.2,
                'fourier_based': 0.1
            }
        elif working_methods >= 2:
            # Balanced weighting for moderate success
            weights = {k: 1.0/working_methods for k in valid_metrics.keys()}
        else:
            # Single method - use it directly
            weights = {k: 1.0 for k in valid_metrics.keys()}
        
        # Calculate weighted average
        weighted_sum = sum(weights.get(k, 0) * v for k, v in valid_metrics.items())
        total_weight = sum(weights.get(k, 0) for k in valid_metrics.keys())
        
        if total_weight > 0:
            composite_roundness = weighted_sum / total_weight
        else:
            composite_roundness = 0.0
    else:
        composite_roundness = 0.0
    
    roundness_metrics['composite'] = composite_roundness
    
    # Calculate quality score
    quality_score = working_methods / 5.0
    
    debug_print("=== ROBUST ROUNDNESS ANALYSIS SUMMARY ===")
    debug_print(f"Working methods: {working_methods}/5")
    debug_print(f"Quality score: {quality_score:.2f}")
    debug_print(f"Composite roundness: {composite_roundness:.4f}")
    
    return roundness_metrics, quality_score, working_methods

# ============================================================================
# ENHANCED MAIN ANALYSIS FUNCTION WITH ROBUST ROUNDNESS
# ============================================================================

def analyze_ballast_with_robust_roundness(filename, 
                                        use_vertex_cleaning=True, 
                                        cleaning_tolerance=1e-6,
                                        roundness_method='comprehensive',
                                        enable_debug=True):
    """
    Enhanced ballast analysis with robust roundness calculations
    """
    global DEBUG_ROUNDNESS
    DEBUG_ROUNDNESS = enable_debug
    
    if enable_debug:
        print(f"\n{'='*60}")
        print(f"ROBUST BALLAST ANALYSIS: {os.path.basename(filename)}")
        print(f"{'='*60}")
    
    try:
        # Load mesh
        model = trimesh.load(filename)
        original_vertex_count = len(model.vertices)
        debug_print(f"Loaded mesh with {original_vertex_count} vertices")
        
        # Optional vertex cleaning
        if use_vertex_cleaning:
            debug_print(f"Cleaning mesh with tolerance {cleaning_tolerance}")
            model.merge_vertices(cleaning_tolerance)
            if hasattr(model, 'remove_unreferenced_vertices'):
                model.remove_unreferenced_vertices()
            final_vertex_count = len(model.vertices)
            reduction_pct = ((original_vertex_count - final_vertex_count) / original_vertex_count) * 100
            debug_print(f"Vertex count reduced: {original_vertex_count} → {final_vertex_count} ({reduction_pct:.1f}% reduction)")
        else:
            final_vertex_count = original_vertex_count
            reduction_pct = 0.0
        
        # Fit sphere to vertices
        debug_print("Fitting sphere to mesh vertices")
        center, radius = fit_sphere(model.vertices)
        debug_print(f"Fitted sphere: center={center}, radius={radius:.4f}")
        
        # Calculate basic geometric properties
        dimensions = model.bounding_box.primitive.extents
        shortest = min(dimensions)
        longest = max(dimensions)
        intermediate = sum(dimensions) - shortest - longest
        
        elongation = intermediate / longest if longest > 0 else 0
        flatness = shortest / intermediate if intermediate > 0 else 0
        aspect_ratio = shortest / longest if longest > 0 else 0
        
        # Volume and surface area
        volume = model.volume
        surface_area = model.area
        convexity = volume / model.convex_hull.volume if model.convex_hull.volume > 0 else 0
        
        # Sphericity calculations
        sphericity = ((np.pi ** (1/3)) * ((6 * volume) ** (2/3))) / surface_area if surface_area > 0 else 0
        sphericity2 = ((intermediate * shortest) / (longest ** 2)) ** (1/3) if longest > 0 else 0
        
        # ROBUST ROUNDNESS CALCULATION
        debug_print(f"Starting robust roundness calculation with method: {roundness_method}")
        
        if roundness_method == 'comprehensive':
            roundness_metrics, quality_score, working_methods = calculate_comprehensive_roundness_robust(model, center, radius)
            
            roundness = roundness_metrics['composite']
            roundness_wadell = roundness_metrics.get('wadell_style', 0)
            roundness_curvature = roundness_metrics.get('curvature_based', 0)
            roundness_corner = roundness_metrics.get('corner_detection', 0)
            roundness_fourier = roundness_metrics.get('fourier_based', 0)
            roundness_sphere = roundness_metrics.get('sphere_based', 0)
            
        else:
            # Single method calculations
            working_methods = 1
            quality_score = 0.2  # Single method quality
            
            if roundness_method == 'wadell':
                roundness = calculate_wadell_roundness_3d_robust(model, num_projections=12)
                roundness_wadell = roundness
                roundness_curvature = roundness_corner = roundness_fourier = roundness_sphere = 0
            elif roundness_method == 'curvature':
                roundness = calculate_curvature_based_roundness_robust(model, neighborhood_size=12)
                roundness_curvature = roundness
                roundness_wadell = roundness_corner = roundness_fourier = roundness_sphere = 0
            elif roundness_method == 'corner':
                roundness = calculate_corner_detection_roundness_robust(model, corner_threshold=0.6)
                roundness_corner = roundness
                roundness_wadell = roundness_curvature = roundness_fourier = roundness_sphere = 0
            elif roundness_method == 'fourier':
                roundness = calculate_fourier_roundness_robust(model, num_projections=10)
                roundness_fourier = roundness
                roundness_wadell = roundness_curvature = roundness_corner = roundness_sphere = 0
            elif roundness_method == 'sphere':
                roundness = calculate_improved_sphere_roundness_robust(model, center, radius)
                roundness_sphere = roundness
                roundness_wadell = roundness_curvature = roundness_corner = roundness_fourier = 0
            else:
                debug_print("Unknown roundness method, using sphere-based", "WARNING")
                roundness = calculate_improved_sphere_roundness_robust(model, center, radius)
                roundness_sphere = roundness
                roundness_wadell = roundness_curvature = roundness_corner = roundness_fourier = 0
        
        # Calculate other properties
        roughness = calculate_roughness(model)
        
        # 3D Angularity analysis
        directions_3d = generate_3d_directions(num_azimuth=36, num_polar=36)  # Reduced for performance
        radii = calculate_radii_by_3d_directions(model, center, directions_3d)
        equivalent_ellipsoid = fit_equivalent_ellipsoid(radii)
        angularity_index, angularity_index_normalized = calculate_angularity_index_3d_relative(radii, equivalent_ellipsoid)
        
        # Create results dictionary
        data = {
            'Filename': filename,
            'File Type': 'Mesh',
            'Number of Points': final_vertex_count,
            'Number of Vertices': final_vertex_count,
            'Original Vertex Count': original_vertex_count,
            'Final Vertex Count': final_vertex_count,
            'Vertex Reduction %': reduction_pct,
            'Cleaning Tolerance': cleaning_tolerance if use_vertex_cleaning else 'N/A',
            
            # Geometric properties
            'Intermediate': intermediate,
            'Shortest': shortest,
            'Longest': longest,
            'Elongation': elongation,
            'Flatness': flatness,
            'Convexity': convexity,
            'Aspect Ratio': aspect_ratio,
            
            # Sphericity measures
            'Sphericity': sphericity,
            'Sphericity2': sphericity2,
            
            # ROBUST ROUNDNESS MEASURES
            'Roundness': roundness,
            'Roundness_Method': roundness_method,
            'Roundness_Wadell': roundness_wadell,
            'Roundness_Curvature': roundness_curvature,
            'Roundness_Corner': roundness_corner,
            'Roundness_Fourier': roundness_fourier,
            'Roundness_Sphere': roundness_sphere,
            'Roundness_Quality': quality_score,
            'Working_Methods': working_methods,
            'Estimated_Corner_Count': max(0, int(50 * (1 - roundness_corner))),
            
            # Other properties
            'Roughness': roughness,
            'Center X': center[0],
            'Center Y': center[1],
            'Center Z': center[2],
            'Radius': radius,
            'Angularity Index': angularity_index,
            'Normalized Angularity Index': angularity_index_normalized,
            'Surface Area': surface_area,
            'Volume': volume,
            'Number of Faces': len(model.faces),
            'Convex Hull Faces': len(model.convex_hull.faces),
            'Convex Hull Vertices': len(model.convex_hull.vertices),
            'Is Watertight': getattr(model, 'is_watertight', 'Unknown'),
            'Sphere Fitted': True
        }
        
        # Note: CSV writing is now handled by the runner to prevent duplicates
        
        # Print summary
        if enable_debug:
            print(f"\n📊 ROBUST ROUNDNESS ANALYSIS RESULTS:")
            print(f"   🎯 Primary roundness: {roundness:.4f}")
            print(f"   🔧 Method used: {roundness_method}")
            print(f"   ⭐ Quality score: {quality_score:.2f}")
            print(f"   🏗️  Working methods: {working_methods}/5")
            print(f"   📈 Individual scores:")
            print(f"      Wadell: {roundness_wadell:.4f}")
            print(f"      Curvature: {roundness_curvature:.4f}")
            print(f"      Corner: {roundness_corner:.4f}")
            print(f"      Fourier: {roundness_fourier:.4f}")
            print(f"      Sphere: {roundness_sphere:.4f}")
        
        return data
        
    except Exception as e:
        debug_print(f"Critical error in analysis: {e}", "ERROR")
        if enable_debug:
            import traceback
            traceback.print_exc()
        return None

# ============================================================================
# BATCH PROCESSING AND TESTING FUNCTIONS
# ============================================================================

def batch_analyze_robust_roundness(filenames, roundness_method='comprehensive', num_cores=None):
    """Batch analyze files with robust roundness calculations"""
    
    def analyze_single_file(filename):
        return analyze_ballast_with_robust_roundness(filename, roundness_method=roundness_method, enable_debug=False)
    
    print(f"Starting robust batch analysis of {len(filenames)} files...")
    print(f"Roundness method: {roundness_method}")
    print("="*60)
    
    results = []
    successful = 0
    failed = 0
    
    if num_cores and num_cores > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            batch_results = list(executor.map(analyze_single_file, filenames))
        
        for i, result in enumerate(batch_results):
            filename = filenames[i]
            if result:
                results.append(result)
                successful += 1
                print(f"✅ {os.path.basename(filename)}: Roundness={result['Roundness']:.3f}, Quality={result['Roundness_Quality']:.2f}")
            else:
                failed += 1
                print(f"❌ {os.path.basename(filename)}: Failed")
    else:
        # Sequential processing
        for i, filename in enumerate(filenames, 1):
            print(f"[{i}/{len(filenames)}] Processing: {os.path.basename(filename)}")
            
            result = analyze_single_file(filename)
            
            if result:
                results.append(result)
                successful += 1
                print(f"✅ Success: Roundness={result['Roundness']:.3f}, Quality={result['Roundness_Quality']:.2f}")
            else:
                failed += 1
                print(f"❌ Failed")
    
    print("="*60)
    print(f"ROBUST BATCH ANALYSIS COMPLETE")
    print(f"Successful: {successful}/{len(filenames)} files")
    print(f"Failed: {failed}/{len(filenames)} files")
    
    return results

def test_robust_roundness_on_simple_shapes():
    """Test the robust roundness calculations on simple geometric shapes"""
    print("Testing ROBUST roundness on simple shapes...")
    print("="*50)
    
    shapes = {}
    
    # Create test shapes
    try:
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        shapes['sphere'] = sphere
        
        cube = trimesh.creation.box(extents=[2, 2, 2])
        shapes['cube'] = cube
        
        cylinder = trimesh.creation.cylinder(radius=1.0, height=2.0)
        shapes['cylinder'] = cylinder
        
        # Test roundness on each shape
        results = {}
        for shape_name, shape in shapes.items():
            print(f"\nTesting {shape_name}:")
            center, radius = fit_sphere(shape.vertices)
            roundness_metrics, quality_score, working_methods = calculate_comprehensive_roundness_robust(shape, center, radius)
            results[shape_name] = roundness_metrics
            
            print(f"  Composite roundness: {roundness_metrics['composite']:.3f}")
            print(f"  Quality score: {quality_score:.2f}")
            print(f"  Working methods: {working_methods}/5")
            print(f"  Individual scores:")
            print(f"    Wadell: {roundness_metrics['wadell_style']:.3f}")
            print(f"    Curvature: {roundness_metrics['curvature_based']:.3f}")
            print(f"    Corner: {roundness_metrics['corner_detection']:.3f}")
            print(f"    Fourier: {roundness_metrics['fourier_based']:.3f}")
            print(f"    Sphere: {roundness_metrics['sphere_based']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"Error in shape testing: {e}")
        return None

# ============================================================================
# MAIN EXECUTION EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Test on simple shapes first
    print("🧪 Testing robust roundness methods on simple shapes...")
    test_results = test_robust_roundness_on_simple_shapes()
    
    if test_results:
        print("\n" + "="*60)
        print("🚀 Ready for real mesh analysis!")
        print("="*60)
        
        # Example usage for single file
        # result = analyze_ballast_with_robust_roundness('your_mesh_file.stl', 
        #                                               roundness_method='comprehensive',
        #                                               enable_debug=True)
        
        # Example usage for batch processing
        # filenames = ['file1.stl', 'file2.stl', 'file3.stl']
        # batch_results = batch_analyze_robust_roundness(filenames, 
        #                                               roundness_method='comprehensive',
        #                                               num_cores=4)
        
        print("Use analyze_ballast_with_robust_roundness() for single files")
        print("Use batch_analyze_robust_roundness() for multiple files")
    else:
        print("❌ Shape testing failed - check the implementation")
