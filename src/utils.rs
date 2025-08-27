//! Utility functions for k-modes clustering

use crate::error::{Error, Result};
use ndarray::{Array1, ArrayView1, ArrayView2};
use std::collections::HashMap;

/// Find the closest centroid for a given data point
pub fn find_closest_centroid<T, F>(
    point: ArrayView1<T>,
    centroids: ArrayView2<T>,
    distance_fn: F,
) -> Result<usize>
where
    T: Clone,
    F: Fn(ArrayView1<T>, ArrayView1<T>) -> Result<f64>,
{
    if centroids.nrows() == 0 {
        return Err(Error::invalid_data("No centroids provided"));
    }

    if centroids.ncols() != point.len() {
        return Err(Error::invalid_data("Point and centroids dimension mismatch"));
    }

    let mut min_distance = f64::INFINITY;
    let mut closest_centroid = 0;

    for (i, centroid) in centroids.rows().into_iter().enumerate() {
        let distance = distance_fn(point, centroid)?;
        if distance < min_distance {
            min_distance = distance;
            closest_centroid = i;
        }
    }

    Ok(closest_centroid)
}

/// Assign all data points to their closest centroids
pub fn assign_points_to_centroids<T, F>(
    data: ArrayView2<T>,
    centroids: ArrayView2<T>,
    distance_fn: F,
) -> Result<Array1<usize>>
where
    T: Clone,
    F: Fn(ArrayView1<T>, ArrayView1<T>) -> Result<f64> + Copy,
{
    let mut assignments = Array1::zeros(data.nrows());

    for (i, point) in data.rows().into_iter().enumerate() {
        assignments[i] = find_closest_centroid(point, centroids, distance_fn)?;
    }

    Ok(assignments)
}

/// Calculate the total cost (sum of distances to centroids) for current assignments
pub fn calculate_cost<T, F>(
    data: ArrayView2<T>,
    centroids: ArrayView2<T>,
    assignments: ArrayView1<usize>,
    distance_fn: F,
) -> Result<f64>
where
    T: Clone,
    F: Fn(ArrayView1<T>, ArrayView1<T>) -> Result<f64>,
{
    let mut total_cost = 0.0;

    for (i, point) in data.rows().into_iter().enumerate() {
        let cluster_id = assignments[i];
        if cluster_id >= centroids.nrows() {
            return Err(Error::invalid_data("Invalid cluster assignment"));
        }
        
        let centroid = centroids.row(cluster_id);
        total_cost += distance_fn(point, centroid)?;
    }

    Ok(total_cost)
}

/// Check if two assignment arrays are equal (for convergence testing)
pub fn assignments_equal(a: ArrayView1<usize>, b: ArrayView1<usize>) -> bool {
    if a.len() != b.len() {
        return false;
    }
    
    a.iter().zip(b.iter()).all(|(&x, &y)| x == y)
}

/// Get indices of points assigned to each cluster
pub fn get_cluster_indices(assignments: ArrayView1<usize>, n_clusters: usize) -> Vec<Vec<usize>> {
    let mut cluster_indices = vec![Vec::new(); n_clusters];
    
    for (point_idx, &cluster_id) in assignments.iter().enumerate() {
        if cluster_id < n_clusters {
            cluster_indices[cluster_id].push(point_idx);
        }
    }
    
    cluster_indices
}

/// Calculate cluster sizes
pub fn cluster_sizes(assignments: ArrayView1<usize>, n_clusters: usize) -> Vec<usize> {
    let mut sizes = vec![0; n_clusters];
    
    for &cluster_id in assignments.iter() {
        if cluster_id < n_clusters {
            sizes[cluster_id] += 1;
        }
    }
    
    sizes
}

/// Calculate inertia (within-cluster sum of squares/distances)
pub fn calculate_inertia<T, F>(
    data: ArrayView2<T>,
    centroids: ArrayView2<T>,
    assignments: ArrayView1<usize>,
    distance_fn: F,
) -> Result<f64>
where
    T: Clone,
    F: Fn(ArrayView1<T>, ArrayView1<T>) -> Result<f64>,
{
    calculate_cost(data, centroids, assignments, distance_fn)
}

/// Validate clustering parameters
pub fn validate_parameters(
    n_clusters: usize,
    max_iter: usize,
    tol: f64,
    n_init: usize,
) -> Result<()> {
    if n_clusters == 0 {
        return Err(Error::invalid_parameter("n_clusters must be > 0"));
    }
    
    if max_iter == 0 {
        return Err(Error::invalid_parameter("max_iter must be > 0"));
    }
    
    if tol < 0.0 {
        return Err(Error::invalid_parameter("tol must be >= 0"));
    }
    
    if n_init == 0 {
        return Err(Error::invalid_parameter("n_init must be > 0"));
    }
    
    Ok(())
}

/// Validate input data
pub fn validate_data<T>(data: ArrayView2<T>) -> Result<()> {
    if data.nrows() == 0 {
        return Err(Error::invalid_data("Data cannot be empty"));
    }
    
    if data.ncols() == 0 {
        return Err(Error::invalid_data("Data must have at least one feature"));
    }
    
    Ok(())
}

/// Simple statistics for categorical features
pub fn categorical_feature_stats<T: Clone + Eq + std::hash::Hash>(
    data: ArrayView2<T>,
    feature_idx: usize,
) -> Result<HashMap<T, usize>> {
    if feature_idx >= data.ncols() {
        return Err(Error::invalid_parameter("Feature index out of bounds"));
    }

    let mut counts = HashMap::new();
    for row in data.rows() {
        let value = row[feature_idx].clone();
        *counts.entry(value).or_insert(0) += 1;
    }

    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::{MatchingDistance, CategoricalDistance};
    use ndarray::Array2;

    #[test]
    fn test_find_closest_centroid() {
        let data = Array2::from_shape_vec((1, 2), vec!["A", "X"]).unwrap();
        let centroids = Array2::from_shape_vec((2, 2), vec!["A", "X", "B", "Y"]).unwrap();
        
        let point = data.row(0);
        let distance = MatchingDistance;
        
        let closest = find_closest_centroid(
            point,
            centroids.view(),
            |a, b| distance.distance(a, b)
        ).unwrap();
        
        assert_eq!(closest, 0); // Should match first centroid exactly
    }

    #[test]
    fn test_assign_points_to_centroids() {
        let data = Array2::from_shape_vec((3, 2), vec!["A", "X", "B", "Y", "A", "X"]).unwrap();
        let centroids = Array2::from_shape_vec((2, 2), vec!["A", "X", "B", "Y"]).unwrap();
        
        let distance = MatchingDistance;
        let assignments = assign_points_to_centroids(
            data.view(),
            centroids.view(),
            |a, b| distance.distance(a, b)
        ).unwrap();
        
        assert_eq!(assignments[0], 0); // A,X -> cluster 0
        assert_eq!(assignments[1], 1); // B,Y -> cluster 1
        assert_eq!(assignments[2], 0); // A,X -> cluster 0
    }

    #[test]
    fn test_calculate_cost() {
        let data = Array2::from_shape_vec((2, 2), vec!["A", "X", "B", "Y"]).unwrap();
        let centroids = Array2::from_shape_vec((2, 2), vec!["A", "X", "B", "Y"]).unwrap();
        let assignments = ndarray::arr1(&[0, 1]);
        
        let distance = MatchingDistance;
        let cost = calculate_cost(
            data.view(),
            centroids.view(),
            assignments.view(),
            |a, b| distance.distance(a, b)
        ).unwrap();
        
        assert_eq!(cost, 0.0); // Perfect assignment, no cost
    }

    #[test]
    fn test_assignments_equal() {
        let a = ndarray::arr1(&[0, 1, 0, 1]);
        let b = ndarray::arr1(&[0, 1, 0, 1]);
        let c = ndarray::arr1(&[1, 0, 1, 0]);
        
        assert!(assignments_equal(a.view(), b.view()));
        assert!(!assignments_equal(a.view(), c.view()));
    }

    #[test]
    fn test_get_cluster_indices() {
        let assignments = ndarray::arr1(&[0, 1, 0, 1, 2]);
        let indices = get_cluster_indices(assignments.view(), 3);
        
        assert_eq!(indices[0], vec![0, 2]);
        assert_eq!(indices[1], vec![1, 3]);
        assert_eq!(indices[2], vec![4]);
    }

    #[test]
    fn test_cluster_sizes() {
        let assignments = ndarray::arr1(&[0, 1, 0, 1, 2]);
        let sizes = cluster_sizes(assignments.view(), 3);
        
        assert_eq!(sizes, vec![2, 2, 1]);
    }

    #[test]
    fn test_validate_parameters() {
        assert!(validate_parameters(2, 100, 0.001, 10).is_ok());
        assert!(validate_parameters(0, 100, 0.001, 10).is_err()); // n_clusters = 0
        assert!(validate_parameters(2, 0, 0.001, 10).is_err());   // max_iter = 0
        assert!(validate_parameters(2, 100, -0.1, 10).is_err()); // negative tol
        assert!(validate_parameters(2, 100, 0.001, 0).is_err()); // n_init = 0
    }

    #[test]
    fn test_validate_data() {
        let good_data = Array2::from_shape_vec((2, 2), vec!["A", "X", "B", "Y"]).unwrap();
        assert!(validate_data(good_data.view()).is_ok());
        
        let empty_data = Array2::from_shape_vec((0, 2), Vec::<&str>::new()).unwrap();
        assert!(validate_data(empty_data.view()).is_err());
    }

    #[test]
    fn test_categorical_feature_stats() {
        let data = Array2::from_shape_vec((4, 2), vec!["A", "X", "A", "Y", "B", "X", "B", "X"]).unwrap();
        
        let stats = categorical_feature_stats(data.view(), 0).unwrap();
        assert_eq!(stats.get("A"), Some(&2));
        assert_eq!(stats.get("B"), Some(&2));
        
        let stats = categorical_feature_stats(data.view(), 1).unwrap();
        assert_eq!(stats.get("X"), Some(&3));
        assert_eq!(stats.get("Y"), Some(&1));
    }
}