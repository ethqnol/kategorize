//! Distance metrics for categorical and mixed data

use crate::error::{Error, Result};
use ndarray::{ArrayView1, ArrayView2};
use std::collections::HashMap;
use std::hash::Hash;

/// Trait for computing distances between categorical data points
pub trait CategoricalDistance<T> {
    /// Compute distance between two categorical data points
    fn distance(&self, a: ArrayView1<T>, b: ArrayView1<T>) -> Result<f64>;

    /// Compute distances between a single point and multiple centroids
    fn distances_to_centroids(&self, point: ArrayView1<T>, centroids: ArrayView2<T>) -> Result<Vec<f64>>;
}

/// Simple matching distance for categorical data
/// Returns 0 if categories match, 1 if they don't
#[derive(Debug, Clone)]
pub struct MatchingDistance;

impl<T: PartialEq> CategoricalDistance<T> for MatchingDistance {
    fn distance(&self, a: ArrayView1<T>, b: ArrayView1<T>) -> Result<f64> {
        if a.len() != b.len() {
            return Err(Error::invalid_data("Vectors must have the same length"));
        }

        let mismatches = a.iter()
            .zip(b.iter())
            .map(|(x, y)| if x == y { 0.0 } else { 1.0 })
            .sum();

        Ok(mismatches)
    }

    fn distances_to_centroids(&self, point: ArrayView1<T>, centroids: ArrayView2<T>) -> Result<Vec<f64>> {
        if centroids.ncols() != point.len() {
            return Err(Error::invalid_data("Point and centroids must have same number of features"));
        }

        let mut distances = Vec::with_capacity(centroids.nrows());
        for centroid_row in centroids.rows() {
            distances.push(self.distance(point, centroid_row)?);
        }
        Ok(distances)
    }
}

/// Hamming distance for categorical data (normalized matching distance)
#[derive(Debug, Clone)]
pub struct HammingDistance;

impl<T: PartialEq> CategoricalDistance<T> for HammingDistance {
    fn distance(&self, a: ArrayView1<T>, b: ArrayView1<T>) -> Result<f64> {
        if a.len() != b.len() {
            return Err(Error::invalid_data("Vectors must have the same length"));
        }

        let mismatches = a.iter()
            .zip(b.iter())
            .map(|(x, y)| if x == y { 0.0 } else { 1.0 })
            .sum::<f64>();

        Ok(mismatches / a.len() as f64)
    }

    fn distances_to_centroids(&self, point: ArrayView1<T>, centroids: ArrayView2<T>) -> Result<Vec<f64>> {
        if centroids.ncols() != point.len() {
            return Err(Error::invalid_data("Point and centroids must have same number of features"));
        }

        let mut distances = Vec::with_capacity(centroids.nrows());
        for centroid_row in centroids.rows() {
            distances.push(self.distance(point, centroid_row)?);
        }
        Ok(distances)
    }
}

/// Jaccard distance for categorical data
/// Measures dissimilarity between two sets based on the size of intersection and union
/// For categorical vectors, treats each vector as a set of unique values
#[derive(Debug, Clone)]
pub struct JaccardDistance;

impl<T: PartialEq + Eq + Hash + Clone> CategoricalDistance<T> for JaccardDistance {
    fn distance(&self, a: ArrayView1<T>, b: ArrayView1<T>) -> Result<f64> {
        if a.len() != b.len() {
            return Err(Error::invalid_data("Vectors must have the same length"));
        }

        // Convert arrays to sets for Jaccard calculation
        let set_a: std::collections::HashSet<_> = a.iter().cloned().collect();
        let set_b: std::collections::HashSet<_> = b.iter().cloned().collect();

        // Calculate intersection and union sizes
        let intersection_size = set_a.intersection(&set_b).count() as f64;
        let union_size = set_a.union(&set_b).count() as f64;

        // Handle edge case where both sets are empty
        if union_size == 0.0 {
            return Ok(0.0); // Two empty sets are identical
        }

        // Jaccard distance = 1 - Jaccard similarity
        // Jaccard similarity = |A ∩ B| / |A ∪ B|
        Ok(1.0 - (intersection_size / union_size))
    }

    fn distances_to_centroids(&self, point: ArrayView1<T>, centroids: ArrayView2<T>) -> Result<Vec<f64>> {
        if centroids.ncols() != point.len() {
            return Err(Error::invalid_data("Point and centroids must have same number of features"));
        }

        let mut distances = Vec::with_capacity(centroids.nrows());
        for centroid_row in centroids.rows() {
            distances.push(self.distance(point, centroid_row)?);
        }
        Ok(distances)
    }
}

/// Euclidean distance for numerical data
#[derive(Debug, Clone)]
pub struct EuclideanDistance;

impl CategoricalDistance<f64> for EuclideanDistance {
    fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> Result<f64> {
        if a.len() != b.len() {
            return Err(Error::invalid_data("Vectors must have the same length"));
        }

        let sum_sq_diff = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>();

        Ok(sum_sq_diff.sqrt())
    }

    fn distances_to_centroids(&self, point: ArrayView1<f64>, centroids: ArrayView2<f64>) -> Result<Vec<f64>> {
        if centroids.ncols() != point.len() {
            return Err(Error::invalid_data("Point and centroids must have same number of features"));
        }

        let mut distances = Vec::with_capacity(centroids.nrows());
        for centroid_row in centroids.rows() {
            distances.push(self.distance(point, centroid_row)?);
        }
        Ok(distances)
    }
}

/// Combined distance for k-prototypes (categorical + numerical data)
#[derive(Debug, Clone)]
pub struct PrototypesDistance {
    categorical_indices: Vec<usize>,
    numerical_indices: Vec<usize>,
    gamma: f64, // Weight for categorical vs numerical features
}

impl PrototypesDistance {
    /// Create a new prototypes distance metric
    pub fn new(categorical_indices: Vec<usize>, numerical_indices: Vec<usize>, gamma: f64) -> Self {
        Self {
            categorical_indices,
            numerical_indices,
            gamma,
        }
    }
}

/// Utility functions for computing modes (most frequent values) in categorical data
pub fn compute_mode<T: Clone + Eq + Hash>(values: &[T]) -> Option<T> {
    if values.is_empty() {
        return None;
    }

    let mut counts = HashMap::new();
    for value in values {
        *counts.entry(value.clone()).or_insert(0) += 1;
    }

    counts.into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(value, _)| value)
}

/// Compute modes for each feature across all data points in a cluster
pub fn compute_modes<T: Clone + Eq + Hash>(
    data: ArrayView2<T>,
    indices: &[usize]
) -> Result<Vec<T>> {
    if indices.is_empty() {
        return Err(Error::invalid_data("Cannot compute mode of empty cluster"));
    }

    let mut modes = Vec::with_capacity(data.ncols());
    
    for col_idx in 0..data.ncols() {
        let column_values: Vec<T> = indices.iter()
            .map(|&row_idx| data[[row_idx, col_idx]].clone())
            .collect();
        
        let mode = compute_mode(&column_values)
            .ok_or_else(|| Error::computation_error("Unable to compute mode for cluster"))?;
        
        modes.push(mode);
    }

    Ok(modes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_matching_distance() {
        let distance = MatchingDistance;
        let a = ndarray::arr1(&["A", "B", "C"]);
        let b = ndarray::arr1(&["A", "X", "C"]);
        
        let result = distance.distance(a.view(), b.view()).unwrap();
        assert_eq!(result, 1.0); // One mismatch
    }

    #[test]
    fn test_hamming_distance() {
        let distance = HammingDistance;
        let a = ndarray::arr1(&["A", "B", "C"]);
        let b = ndarray::arr1(&["A", "X", "C"]);
        
        let result = distance.distance(a.view(), b.view()).unwrap();
        assert!((result - 1.0/3.0).abs() < 1e-10); // One mismatch out of 3
    }

    #[test]
    fn test_euclidean_distance() {
        let distance = EuclideanDistance;
        let a = ndarray::arr1(&[1.0, 2.0, 3.0]);
        let b = ndarray::arr1(&[4.0, 5.0, 6.0]);
        
        let result = distance.distance(a.view(), b.view()).unwrap();
        let expected = ((3.0_f64).powi(2) * 3.0).sqrt();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_compute_mode() {
        let values = vec!["A", "B", "A", "C", "A"];
        let mode = compute_mode(&values).unwrap();
        assert_eq!(mode, "A");
    }

    #[test]
    fn test_compute_modes() {
        let data = Array2::from_shape_vec((3, 2), vec!["A", "X", "B", "Y", "A", "X"]).unwrap();
        let indices = vec![0, 2]; // Rows 0 and 2
        
        let modes = compute_modes(data.view(), &indices).unwrap();
        assert_eq!(modes, vec!["A", "X"]);
    }

    #[test]
    fn test_jaccard_distance_identical() {
        let distance = JaccardDistance;
        let a = ndarray::arr1(&["A", "B", "C"]);
        let b = ndarray::arr1(&["A", "B", "C"]);
        
        let result = distance.distance(a.view(), b.view()).unwrap();
        assert_eq!(result, 0.0); // Identical sets should have distance 0
    }

    #[test]
    fn test_jaccard_distance_disjoint() {
        let distance = JaccardDistance;
        let a = ndarray::arr1(&["A", "B", "C"]);
        let b = ndarray::arr1(&["X", "Y", "Z"]);
        
        let result = distance.distance(a.view(), b.view()).unwrap();
        assert_eq!(result, 1.0); // Disjoint sets should have distance 1
    }

    #[test]
    fn test_jaccard_distance_partial_overlap() {
        let distance = JaccardDistance;
        let a = ndarray::arr1(&["A", "B", "C"]);
        let b = ndarray::arr1(&["B", "C", "D"]);
        
        let result = distance.distance(a.view(), b.view()).unwrap();
        // Intersection: {B, C} (size 2)
        // Union: {A, B, C, D} (size 4)
        // Jaccard similarity = 2/4 = 0.5
        // Jaccard distance = 1 - 0.5 = 0.5
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_distance_with_duplicates() {
        let distance = JaccardDistance;
        let a = ndarray::arr1(&["A", "A", "B", "B"]);
        let b = ndarray::arr1(&["A", "B", "B", "C"]);
        
        let result = distance.distance(a.view(), b.view()).unwrap();
        // Set A: {A, B} (size 2)
        // Set B: {A, B, C} (size 3)
        // Intersection: {A, B} (size 2)
        // Union: {A, B, C} (size 3)
        // Jaccard similarity = 2/3
        // Jaccard distance = 1 - 2/3 = 1/3
        assert!((result - 1.0/3.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_distance_empty_sets() {
        let distance = JaccardDistance;
        let a = ndarray::arr1(&[] as &[&str]);
        let b = ndarray::arr1(&[] as &[&str]);
        
        let result = distance.distance(a.view(), b.view()).unwrap();
        assert_eq!(result, 0.0); // Two empty sets are identical
    }

    #[test]
    fn test_jaccard_distance_centroids() {
        let distance = JaccardDistance;
        let point = ndarray::arr1(&["A", "B", "C"]);
        let centroids = Array2::from_shape_vec(
            (2, 3), 
            vec!["A", "B", "C", "X", "Y", "Z"]
        ).unwrap();
        
        let distances = distance.distances_to_centroids(point.view(), centroids.view()).unwrap();
        assert_eq!(distances.len(), 2);
        assert_eq!(distances[0], 0.0); // Identical to first centroid
        assert_eq!(distances[1], 1.0); // Disjoint from second centroid
    }
}