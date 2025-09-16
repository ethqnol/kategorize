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

/// Frequency tracker for incremental mode computation
#[derive(Debug, Clone)]
pub struct FrequencyTracker<T: Clone + Eq + Hash> {
    counts: HashMap<T, usize>,
    mode: Option<T>,
    mode_count: usize,
    total_count: usize,
}

impl<T: Clone + Eq + Hash> FrequencyTracker<T> {
    /// Create a new frequency tracker
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            mode: None,
            mode_count: 0,
            total_count: 0,
        }
    }

    /// Add a value to the frequency tracker
    pub fn add(&mut self, value: &T) {
        let new_count = *self.counts.get(value).unwrap_or(&0) + 1;
        self.counts.insert(value.clone(), new_count);
        self.total_count += 1;

        // Update mode if this value now has the highest count
        if new_count > self.mode_count {
            self.mode = Some(value.clone());
            self.mode_count = new_count;
        }
    }

    /// Remove a value from the frequency tracker
    pub fn remove(&mut self, value: &T) -> Result<()> {
        let current_count = self.counts.get(value).copied().unwrap_or(0);
        
        if current_count == 0 {
            return Err(Error::computation_error("Cannot remove value not in tracker"));
        }

        self.total_count -= 1;
        
        if current_count == 1 {
            self.counts.remove(value);
        } else {
            self.counts.insert(value.clone(), current_count - 1);
        }

        // If we removed the mode, we need to find the new mode
        if self.mode.as_ref() == Some(value) && current_count == self.mode_count {
            self.recompute_mode();
        }

        Ok(())
    }

    /// Get the current mode
    pub fn mode(&self) -> Option<&T> {
        self.mode.as_ref()
    }

    /// Clear all frequencies
    pub fn clear(&mut self) {
        self.counts.clear();
        self.mode = None;
        self.mode_count = 0;
        self.total_count = 0;
    }

    /// Initialize with a set of values
    pub fn init_with_values(&mut self, values: &[T]) {
        self.clear();
        for value in values {
            self.add(value);
        }
    }

    /// Recompute the mode from scratch (used when mode is invalidated)
    fn recompute_mode(&mut self) {
        if self.counts.is_empty() {
            self.mode = None;
            self.mode_count = 0;
            return;
        }

        let (mode_value, mode_count) = self.counts.iter()
            .max_by_key(|(_, count)| *count)
            .map(|(value, count)| (value.clone(), *count))
            .unwrap();
        
        self.mode = Some(mode_value);
        self.mode_count = mode_count;
    }
}

impl<T: Clone + Eq + Hash> Default for FrequencyTracker<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for computing modes (most frequent values) in categorical data
pub fn compute_mode<T: Clone + Eq + Hash>(values: &[T]) -> Option<T> {
    if values.is_empty() {
        return None;
    }

    let mut tracker = FrequencyTracker::new();
    tracker.init_with_values(values);
    tracker.mode().cloned()
}

/// Centroid tracker for efficient incremental mode computation
#[derive(Debug, Clone)]
pub struct CentroidTracker<T: Clone + Eq + Hash> {
    feature_trackers: Vec<FrequencyTracker<T>>,
    point_assignments: HashMap<usize, Vec<T>>, // Maps point index to its feature values
}

impl<T: Clone + Eq + Hash> CentroidTracker<T> {
    /// Create a new centroid tracker with the given number of features
    pub fn new(num_features: usize) -> Self {
        Self {
            feature_trackers: vec![FrequencyTracker::new(); num_features],
            point_assignments: HashMap::new(),
        }
    }

    /// Add a data point to this centroid
    pub fn add_point(&mut self, point_idx: usize, values: &[T]) -> Result<()> {
        if values.len() != self.feature_trackers.len() {
            return Err(Error::invalid_data("Point has wrong number of features"));
        }

        // Remove previous assignment if it exists
        if let Some(old_values) = self.point_assignments.get(&point_idx) {
            for (tracker, old_value) in self.feature_trackers.iter_mut().zip(old_values.iter()) {
                tracker.remove(old_value)?;
            }
        }

        // Add new assignment
        for (tracker, value) in self.feature_trackers.iter_mut().zip(values.iter()) {
            tracker.add(value);
        }

        self.point_assignments.insert(point_idx, values.to_vec());
        Ok(())
    }

    /// Remove a data point from this centroid
    pub fn remove_point(&mut self, point_idx: usize) -> Result<()> {
        if let Some(values) = self.point_assignments.remove(&point_idx) {
            for (tracker, value) in self.feature_trackers.iter_mut().zip(values.iter()) {
                tracker.remove(value)?;
            }
        }
        Ok(())
    }

    /// Get the current centroid (modes for each feature)
    pub fn get_centroid(&self) -> Result<Vec<T>> {
        let mut centroid = Vec::with_capacity(self.feature_trackers.len());
        
        for tracker in &self.feature_trackers {
            if let Some(mode) = tracker.mode() {
                centroid.push(mode.clone());
            } else {
                return Err(Error::computation_error("No mode available for feature"));
            }
        }
        
        Ok(centroid)
    }

    /// Check if the centroid is empty (no points assigned)
    pub fn is_empty(&self) -> bool {
        self.point_assignments.is_empty()
    }

    /// Clear all assignments
    pub fn clear(&mut self) {
        for tracker in &mut self.feature_trackers {
            tracker.clear();
        }
        self.point_assignments.clear();
    }

    /// Initialize with a set of data points
    pub fn init_with_points<I>(&mut self, data: ArrayView2<T>, point_indices: I) -> Result<()>
    where
        I: IntoIterator<Item = usize>,
    {
        self.clear();
        
        for point_idx in point_indices {
            if point_idx >= data.nrows() {
                return Err(Error::invalid_data("Point index out of bounds"));
            }
            
            let point_values: Vec<T> = (0..data.ncols())
                .map(|col| data[[point_idx, col]].clone())
                .collect();
            
            self.add_point(point_idx, &point_values)?;
        }
        
        Ok(())
    }
}

/// Compute modes for each feature across all data points in a cluster
pub fn compute_modes<T: Clone + Eq + Hash>(
    data: ArrayView2<T>,
    indices: &[usize]
) -> Result<Vec<T>> {
    if indices.is_empty() {
        return Err(Error::invalid_data("Cannot compute mode of empty cluster"));
    }

    let mut tracker = CentroidTracker::new(data.ncols());
    tracker.init_with_points(data, indices.iter().copied())?;
    tracker.get_centroid()
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

    #[test]
    fn test_frequency_tracker() {
        let mut tracker = FrequencyTracker::new();
        
        // Test empty tracker
        assert_eq!(tracker.mode(), None);
        
        // Add values
        tracker.add(&"A");
        assert_eq!(tracker.mode(), Some(&"A"));
        
        tracker.add(&"B");
        tracker.add(&"A");
        assert_eq!(tracker.mode(), Some(&"A")); // A appears twice, B once
        
        // Remove a value
        tracker.remove(&"A").unwrap();
        // Now A and B both appear once, but A was added first so it should still be mode
        // (depending on HashMap iteration order, but it should be consistent)
        assert!(tracker.mode().is_some());
        
        // Add more B's to make B the mode
        tracker.add(&"B");
        assert_eq!(tracker.mode(), Some(&"B")); // B now appears twice
    }

    #[test]
    fn test_frequency_tracker_init_with_values() {
        let mut tracker = FrequencyTracker::new();
        let values = vec!["A", "B", "A", "C", "A"];
        tracker.init_with_values(&values);
        
        assert_eq!(tracker.mode(), Some(&"A")); // A appears 3 times
    }

    #[test]
    fn test_centroid_tracker() {
        let data = Array2::from_shape_vec((3, 2), vec!["A", "X", "B", "Y", "A", "X"]).unwrap();
        let mut tracker = CentroidTracker::new(2);
        
        // Add points
        tracker.add_point(0, &["A", "X"]).unwrap();
        tracker.add_point(2, &["A", "X"]).unwrap();
        
        let centroid = tracker.get_centroid().unwrap();
        assert_eq!(centroid, vec!["A", "X"]);
        
        // Remove a point
        tracker.remove_point(0).unwrap();
        let centroid = tracker.get_centroid().unwrap();
        assert_eq!(centroid, vec!["A", "X"]); // Still the same since both points had same values
        
        // Add a different point
        tracker.add_point(1, &["B", "Y"]).unwrap();
        // Now we have two different values, the mode should be consistent
        let centroid = tracker.get_centroid().unwrap();
        assert_eq!(centroid.len(), 2);
    }

    #[test]
    fn test_centroid_tracker_init_with_points() {
        let data = Array2::from_shape_vec((4, 2), vec!["A", "X", "A", "X", "B", "Y", "B", "Y"]).unwrap();
        let mut tracker = CentroidTracker::new(2);
        
        // Initialize with points 0 and 1 (both have ["A", "X"])
        tracker.init_with_points(data.view(), vec![0, 1]).unwrap();
        
        let centroid = tracker.get_centroid().unwrap();
        assert_eq!(centroid, vec!["A", "X"]);
        
        // Clear and initialize with all points
        tracker.clear();
        tracker.init_with_points(data.view(), vec![0, 1, 2, 3]).unwrap();
        
        // Should have mixed modes now
        let centroid = tracker.get_centroid().unwrap();
        assert_eq!(centroid.len(), 2);
    }
}