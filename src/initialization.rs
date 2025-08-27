//! Initialization methods for k-modes and k-prototypes clustering

use crate::distance::{CategoricalDistance, MatchingDistance};
use crate::error::{Error, Result};
use ndarray::{Array2, ArrayView2};
use rand::prelude::*;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Initialization methods for clustering algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum InitMethod {
    /// Random initialization - randomly select data points as initial centroids
    Random,
    /// Huang initialization - density-based initialization method
    Huang,
    /// Cao initialization - improved initialization based on Cao et al.
    Cao,
}

/// Initialize centroids for k-modes clustering
pub fn initialize_centroids<T, R>(
    data: ArrayView2<T>,
    n_clusters: usize,
    method: InitMethod,
    rng: &mut R,
) -> Result<Array2<T>>
where
    T: Clone + Eq + Hash,
    R: Rng,
{
    if n_clusters == 0 {
        return Err(Error::invalid_parameter("Number of clusters must be > 0"));
    }
    
    if n_clusters > data.nrows() {
        return Err(Error::invalid_parameter(
            "Number of clusters cannot exceed number of data points"
        ));
    }

    match method {
        InitMethod::Random => random_init(data, n_clusters, rng),
        InitMethod::Huang => huang_init(data, n_clusters, rng),
        InitMethod::Cao => cao_init(data, n_clusters, rng),
    }
}

/// Random initialization: randomly select k data points as initial centroids
fn random_init<T, R>(
    data: ArrayView2<T>,
    n_clusters: usize,
    rng: &mut R,
) -> Result<Array2<T>>
where
    T: Clone,
    R: Rng,
{
    let mut selected_indices = HashSet::new();
    let n_points = data.nrows();

    // Randomly select unique indices
    while selected_indices.len() < n_clusters {
        let idx = rng.gen_range(0..n_points);
        selected_indices.insert(idx);
    }

    let indices: Vec<_> = selected_indices.into_iter().collect();
    let mut centroids = Array2::uninit((n_clusters, data.ncols()));

    for (i, &data_idx) in indices.iter().enumerate() {
        for j in 0..data.ncols() {
            centroids[[i, j]].write(data[[data_idx, j]].clone());
        }
    }

    // Safety: we've initialized all elements
    Ok(unsafe { centroids.assume_init() })
}

/// Huang initialization: density-based initialization
/// Selects points with highest density (frequency) as initial centroids
fn huang_init<T, R>(
    data: ArrayView2<T>,
    n_clusters: usize,
    rng: &mut R,
) -> Result<Array2<T>>
where
    T: Clone + Eq + Hash,
    R: Rng,
{
    // Calculate frequency of each unique data point
    let mut point_frequencies = HashMap::new();
    for row in data.rows() {
        let point: Vec<T> = row.iter().cloned().collect();
        *point_frequencies.entry(point).or_insert(0) += 1;
    }

    // Sort by frequency (descending)
    let mut sorted_points: Vec<_> = point_frequencies.into_iter().collect();
    sorted_points.sort_by(|a, b| b.1.cmp(&a.1));

    // If we have fewer unique points than clusters, fall back to random
    if sorted_points.len() < n_clusters {
        return random_init(data, n_clusters, rng);
    }

    // Select the most frequent points as centroids
    let mut centroids = Array2::uninit((n_clusters, data.ncols()));
    for (i, (point, _)) in sorted_points.iter().take(n_clusters).enumerate() {
        for (j, value) in point.iter().enumerate() {
            centroids[[i, j]].write(value.clone());
        }
    }

    // Safety: we've initialized all elements
    Ok(unsafe { centroids.assume_init() })
}

/// Cao initialization: improved density-based method
/// Uses both density and dissimilarity to select diverse initial centroids
fn cao_init<T, R>(
    data: ArrayView2<T>,
    n_clusters: usize,
    rng: &mut R,
) -> Result<Array2<T>>
where
    T: Clone + Eq + Hash,
    R: Rng,
{
    let distance_metric = MatchingDistance;
    
    // First, select the most frequent point
    let mut point_frequencies = HashMap::new();
    let mut point_to_idx = HashMap::new();
    
    for (idx, row) in data.rows().into_iter().enumerate() {
        let point: Vec<T> = row.iter().cloned().collect();
        *point_frequencies.entry(point.clone()).or_insert(0) += 1;
        point_to_idx.entry(point).or_insert(idx);
    }

    let first_centroid_point = point_frequencies
        .iter()
        .max_by_key(|(_, &freq)| freq)
        .map(|(point, _)| point.clone())
        .ok_or_else(|| Error::initialization_failure("No data points found"))?;

    let mut selected_points = vec![first_centroid_point];
    let mut selected_indices = HashSet::new();
    if let Some(&idx) = point_to_idx.get(&selected_points[0]) {
        selected_indices.insert(idx);
    }

    // For remaining centroids, select points that maximize distance to existing centroids
    while selected_points.len() < n_clusters {
        let mut best_point: Option<Vec<T>> = None;
        let mut best_distance = -1.0;

        for (point, _) in &point_frequencies {
            if selected_points.contains(point) {
                continue;
            }

            // Calculate minimum distance to existing centroids
            let point_array = ndarray::Array1::from_vec(point.clone());
            let mut min_distance = f64::INFINITY;

            for selected_point in &selected_points {
                let selected_array = ndarray::Array1::from_vec(selected_point.clone());
                let dist = distance_metric.distance(point_array.view(), selected_array.view())?;
                min_distance = min_distance.min(dist);
            }

            if min_distance > best_distance {
                best_distance = min_distance;
                best_point = Some(point.clone());
            }
        }

        match best_point {
            Some(point) => selected_points.push(point),
            None => break, // Fallback: not enough diverse points
        }
    }

    // If we couldn't find enough diverse points, fill with random selection
    while selected_points.len() < n_clusters {
        let available_points: Vec<_> = point_frequencies
            .keys()
            .filter(|p| !selected_points.contains(p))
            .collect();
        
        if available_points.is_empty() {
            return Err(Error::initialization_failure("Insufficient unique data points"));
        }

        let random_point = available_points[rng.gen_range(0..available_points.len())].clone();
        selected_points.push(random_point);
    }

    // Build centroids array
    let mut centroids = Array2::uninit((n_clusters, data.ncols()));
    for (i, point) in selected_points.iter().enumerate() {
        for (j, value) in point.iter().enumerate() {
            centroids[[i, j]].write(value.clone());
        }
    }

    // Safety: we've initialized all elements
    Ok(unsafe { centroids.assume_init() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_random_init() {
        let data = Array2::from_shape_vec((4, 2), vec!["A", "X", "B", "Y", "A", "X", "C", "Z"]).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        
        let centroids = random_init(data.view(), 2, &mut rng).unwrap();
        assert_eq!(centroids.dim(), (2, 2));
    }

    #[test]
    fn test_huang_init() {
        let data = Array2::from_shape_vec((4, 2), vec!["A", "X", "A", "X", "B", "Y", "C", "Z"]).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        
        let centroids = huang_init(data.view(), 2, &mut rng).unwrap();
        assert_eq!(centroids.dim(), (2, 2));
        // The most frequent point should be selected first
        assert_eq!(centroids.row(0), ndarray::arr1(&["A", "X"]).view());
    }

    #[test]
    fn test_cao_init() {
        let data = Array2::from_shape_vec((4, 2), vec!["A", "X", "A", "X", "B", "Y", "C", "Z"]).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        
        let centroids = cao_init(data.view(), 2, &mut rng).unwrap();
        assert_eq!(centroids.dim(), (2, 2));
    }

    #[test]
    fn test_initialize_centroids() {
        let data = Array2::from_shape_vec((4, 2), vec!["A", "X", "B", "Y", "A", "X", "C", "Z"]).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        for method in [InitMethod::Random, InitMethod::Huang, InitMethod::Cao] {
            let centroids = initialize_centroids(data.view(), 2, method, &mut rng).unwrap();
            assert_eq!(centroids.dim(), (2, 2));
        }
    }

    #[test]
    fn test_invalid_parameters() {
        let data = Array2::from_shape_vec((2, 2), vec!["A", "X", "B", "Y"]).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        // Test zero clusters
        assert!(initialize_centroids(data.view(), 0, InitMethod::Random, &mut rng).is_err());

        // Test more clusters than data points
        assert!(initialize_centroids(data.view(), 3, InitMethod::Random, &mut rng).is_err());
    }
}