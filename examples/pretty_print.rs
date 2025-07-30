use kll2::Sketch;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn main() {
    // Example 1: Numeric sketch with uniform distribution
    println!("Example 1: Uniform distribution (0-1000)");
    println!("========================================\n");

    let mut sketch = Sketch::new(100);
    let mut rng = SmallRng::from_os_rng();

    for _ in 0..10_000 {
        sketch.update(rng.random_range(0.0..1000.0));
    }

    println!("{}", sketch);

    // Show histogram of the uniform distribution
    println!("\nHistogram view:");
    println!("{}", sketch.histogram(10));

    // Example 2: Normal distribution
    println!("\nExample 2: Normal distribution (μ=50, σ=10)");
    println!("============================================\n");

    let mut normal_sketch = Sketch::new(75);
    use rand_distr::{Distribution, Normal};
    let normal = Normal::new(50.0, 10.0).unwrap();

    for _ in 0..5_000 {
        normal_sketch.update(normal.sample(&mut rng));
    }

    println!("{}", normal_sketch);

    // Show histogram of the normal distribution
    println!("\nHistogram view (Normal distribution):");
    println!("{}", normal_sketch.histogram(15));

    // Example 3: String sketch
    println!("\nExample 3: String sketch with city names");
    println!("========================================\n");

    let mut string_sketch = Sketch::<String>::new(50);
    let cities = vec![
        "New York",
        "Los Angeles",
        "Chicago",
        "Houston",
        "Phoenix",
        "Philadelphia",
        "San Antonio",
        "San Diego",
        "Dallas",
        "San Jose",
    ];

    // Add cities with different frequencies
    for (i, city) in cities.iter().enumerate() {
        let frequency = (i + 1) * 10; // NYC appears 10 times, LA 20 times, etc.
        for _ in 0..frequency {
            string_sketch.update(city.to_string());
        }
    }

    println!("{}", string_sketch);

    // Example 4: Small sketch to show compaction in action
    println!("\nExample 4: Small sketch (k=10) showing compaction");
    println!("=================================================\n");

    let mut small_sketch = Sketch::new(10);

    for i in 0..100 {
        small_sketch.update(i as f64);
        if i == 20 || i == 50 || i == 99 {
            println!("After {} updates:", i + 1);
            println!("{}", small_sketch);
        }
    }

    // Example 5: Vec<u8> sketch
    println!("\nExample 5: Vec<u8> sketch with byte arrays");
    println!("==========================================\n");

    let mut bytes_sketch = Sketch::<Vec<u8>>::new(30);

    // Add some byte arrays
    bytes_sketch.update(vec![0x01, 0x02, 0x03]);
    bytes_sketch.update(vec![0x10, 0x20, 0x30, 0x40]);
    bytes_sketch.update(vec![0xff, 0xfe, 0xfd]);

    // Add longer byte arrays
    for i in 0..20 {
        let mut bytes = vec![i as u8; (i % 10) + 1];
        bytes[0] = (i * 10) as u8;
        bytes_sketch.update(bytes);
    }

    println!("{}", bytes_sketch);

    // Example 6: Integer sketch
    println!("\nExample 6: Integer sketch");
    println!("========================\n");

    let mut int_sketch = Sketch::<i32>::new(50);

    for i in -50..50 {
        int_sketch.update(i * i);
    }

    println!("{}", int_sketch);
}
