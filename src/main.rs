use kll2::Sketch;

fn main() {
    // Example 1: Using with f64 (floating point numbers)
    let mut float_sketch = Sketch::<f64>::new(100);
    for i in 0..1000 {
        float_sketch.update(i as f64);
    }
    println!("Float sketch count: {}", float_sketch.count());
    println!("Float sketch rank of 500.0: {}", float_sketch.rank(500.0));

    // Example 2: Using with i32 (integers)
    let mut int_sketch = Sketch::<i32>::new(100);
    for i in 0..1000 {
        int_sketch.update(i);
    }
    println!("\nInteger sketch count: {}", int_sketch.count());
    println!("Integer sketch rank of 500: {}", int_sketch.rank(500));

    // Example 3: Using with String
    let mut string_sketch = Sketch::<String>::new(50);
    let words = vec!["apple", "banana", "cherry", "date", "elderberry"];
    for word in &words {
        for _ in 0..10 {
            string_sketch.update(word.to_string());
        }
    }
    println!("\nString sketch count: {}", string_sketch.count());
    println!(
        "String sketch rank of 'cherry': {}",
        string_sketch.rank("cherry".to_string())
    );

    // Example 4: Using with Vec<u8> (byte arrays)
    let mut byte_sketch = Sketch::<Vec<u8>>::new(50);
    for i in 0..100 {
        byte_sketch.update(vec![i as u8, (i * 2) as u8]);
    }
    println!("\nByte array sketch count: {}", byte_sketch.count());

    // Example 5: Demonstrating CDF functionality
    let cdf = float_sketch.cdf();
    println!("\nCDF query at 0.5: {}", cdf.query(0.5));
    println!("CDF quantile at 500.0: {}", cdf.quantile(500.0));
}
