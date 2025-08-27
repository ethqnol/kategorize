use kategorize::{KModes, InitMethod};
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::Array2;
use rand::prelude::*;

fn generate_categorical_data(n_samples: usize, n_features: usize, n_categories: usize) -> Array2<String> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut data = Vec::with_capacity(n_samples * n_features);
    
    for _ in 0..n_samples {
        for _ in 0..n_features {
            let category = rng.gen_range(0..n_categories);
            data.push(format!("cat_{}", category));
        }
    }
    
    Array2::from_shape_vec((n_samples, n_features), data).unwrap()
}

fn bench_kmodes_small(c: &mut Criterion) {
    let data = generate_categorical_data(100, 5, 10);
    
    let mut group = c.benchmark_group("kmodes_small");
    
    for &n_clusters in &[2, 5, 10] {
        group.bench_with_input(
            BenchmarkId::new("huang_init", n_clusters),
            &n_clusters,
            |b, &k| {
                let kmodes = KModes::new(k)
                    .init_method(InitMethod::Huang)
                    .random_state(42)
                    .n_init(1)
                    .max_iter(50);
                
                b.iter(|| {
                    black_box(kmodes.fit(black_box(data.view())).unwrap())
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("random_init", n_clusters),
            &n_clusters,
            |b, &k| {
                let kmodes = KModes::new(k)
                    .init_method(InitMethod::Random)
                    .random_state(42)
                    .n_init(1)
                    .max_iter(50);
                
                b.iter(|| {
                    black_box(kmodes.fit(black_box(data.view())).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_kmodes_medium(c: &mut Criterion) {
    let data = generate_categorical_data(1000, 10, 20);
    
    let mut group = c.benchmark_group("kmodes_medium");
    group.sample_size(20); // Fewer samples for longer benchmarks
    
    for &n_clusters in &[5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new("huang_init", n_clusters),
            &n_clusters,
            |b, &k| {
                let kmodes = KModes::new(k)
                    .init_method(InitMethod::Huang)
                    .random_state(42)
                    .n_init(1)
                    .max_iter(100);
                
                b.iter(|| {
                    black_box(kmodes.fit(black_box(data.view())).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_kmodes_large(c: &mut Criterion) {
    let data = generate_categorical_data(5000, 15, 30);
    
    let mut group = c.benchmark_group("kmodes_large");
    group.sample_size(10); // Even fewer samples for very long benchmarks
    
    for &n_clusters in &[10, 20] {
        group.bench_with_input(
            BenchmarkId::new("huang_init", n_clusters),
            &n_clusters,
            |b, &k| {
                let kmodes = KModes::new(k)
                    .init_method(InitMethod::Huang)
                    .random_state(42)
                    .n_init(1)
                    .max_iter(50);
                
                b.iter(|| {
                    black_box(kmodes.fit(black_box(data.view())).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_initialization_methods(c: &mut Criterion) {
    let data = generate_categorical_data(500, 8, 15);
    
    let mut group = c.benchmark_group("initialization_methods");
    
    let methods = [
        ("huang", InitMethod::Huang),
        ("random", InitMethod::Random),
        ("cao", InitMethod::Cao),
    ];
    
    for (name, method) in &methods {
        group.bench_function(*name, |b| {
            let kmodes = KModes::new(8)
                .init_method(*method)
                .random_state(42)
                .n_init(1)
                .max_iter(50);
            
            b.iter(|| {
                black_box(kmodes.fit(black_box(data.view())).unwrap())
            });
        });
    }
    
    group.finish();
}

fn bench_n_init_effect(c: &mut Criterion) {
    let data = generate_categorical_data(300, 6, 12);
    
    let mut group = c.benchmark_group("n_init_effect");
    
    for &n_init in &[1, 3, 5, 10] {
        group.bench_with_input(
            BenchmarkId::from_parameter(n_init),
            &n_init,
            |b, &n_init| {
                let kmodes = KModes::new(5)
                    .init_method(InitMethod::Huang)
                    .random_state(42)
                    .n_init(n_init)
                    .max_iter(50);
                
                b.iter(|| {
                    black_box(kmodes.fit(black_box(data.view())).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_data_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_size_scaling");
    
    let sizes = [
        (50, 3, 5),    // (samples, features, categories)
        (100, 5, 8),
        (200, 7, 10),
        (500, 10, 15),
    ];
    
    for &(n_samples, n_features, n_categories) in &sizes {
        let data = generate_categorical_data(n_samples, n_features, n_categories);
        
        group.bench_with_input(
            BenchmarkId::new("scaling", format!("{}x{}x{}", n_samples, n_features, n_categories)),
            &data,
            |b, data| {
                let kmodes = KModes::new(5)
                    .init_method(InitMethod::Huang)
                    .random_state(42)
                    .n_init(1)
                    .max_iter(50);
                
                b.iter(|| {
                    black_box(kmodes.fit(black_box(data.view())).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_kmodes_small,
    bench_kmodes_medium,
    bench_kmodes_large,
    bench_initialization_methods,
    bench_n_init_effect,
    bench_data_size_scaling
);
criterion_main!(benches);