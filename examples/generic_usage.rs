use kll2::Sketch;

fn main() {
    println!("=== KLL Sketch Generic Usage Examples ===\n");

    // Example 1: Integer sketch
    println!("1. Integer Sketch (i32)");
    let mut int_sketch = Sketch::<i32>::new(100);
    for i in 0..1000 {
        int_sketch.update(i);
    }
    println!("  Count: {}", int_sketch.count());
    println!("  Rank of 500: {}", int_sketch.rank(500));
    let int_cdf = int_sketch.cdf();
    println!("  Median: {}", int_cdf.query(0.5));
    println!("  90th percentile: {}\n", int_cdf.query(0.9));

    // Example 2: String sketch
    println!("2. String Sketch");
    let mut string_sketch = Sketch::<String>::new(50);
    let words = vec!["apple", "banana", "cherry", "date", "elderberry"];
    for word in &words {
        for _ in 0..20 {
            string_sketch.update(word.to_string());
        }
    }
    println!("  Count: {}", string_sketch.count());
    println!(
        "  Rank of 'cherry': {}",
        string_sketch.rank("cherry".to_string())
    );
    let str_cdf = string_sketch.cdf();
    println!("  Median word: {}", str_cdf.query(0.5));
    println!("  First quartile: {}\n", str_cdf.query(0.25));

    // Example 3: Vec<u8> sketch (byte arrays)
    println!("3. Byte Array Sketch (Vec<u8>)");
    let mut byte_sketch = Sketch::<Vec<u8>>::new(100);

    // Add different length byte arrays
    for i in 0..50 {
        byte_sketch.update(vec![i as u8]); // 1-byte arrays
        byte_sketch.update(vec![i as u8, i as u8]); // 2-byte arrays
        byte_sketch.update(vec![i as u8, i as u8, i as u8]); // 3-byte arrays
    }

    println!("  Count: {}", byte_sketch.count());
    println!("  Rank of [25]: {}", byte_sketch.rank(vec![25]));
    println!("  Rank of [25, 25]: {}", byte_sketch.rank(vec![25, 25]));

    let byte_cdf = byte_sketch.cdf();
    let median_bytes = byte_cdf.query(0.5);
    println!("  Median byte array: {:?}", median_bytes);
    println!("  Median length: {} bytes\n", median_bytes.len());

    // Example 4: Different integer types
    println!("4. Different Integer Types");

    let mut u64_sketch = Sketch::<u64>::new(50);
    let mut i64_sketch = Sketch::<i64>::new(50);

    for i in 0..100 {
        u64_sketch.update(i as u64);
        i64_sketch.update(i as i64 - 50); // Include negative numbers
    }

    println!(
        "  u64 sketch - Count: {}, Median: {}",
        u64_sketch.count(),
        u64_sketch.cdf().query(0.5)
    );

    println!(
        "  i64 sketch - Count: {}, Median: {}",
        i64_sketch.count(),
        i64_sketch.cdf().query(0.5)
    );

    // Example 5: Merging sketches
    println!("\n5. Merging String Sketches");
    let mut sketch1 = Sketch::<String>::new(50);
    let mut sketch2 = Sketch::<String>::new(50);

    // Add different data to each sketch
    for _ in 0..50 {
        sketch1.update("morning".to_string());
        sketch2.update("evening".to_string());
    }

    println!("  Sketch 1 count before merge: {}", sketch1.count());
    println!("  Sketch 2 count: {}", sketch2.count());

    sketch1.merge(&sketch2);

    println!("  Sketch 1 count after merge: {}", sketch1.count());
    let merged_cdf = sketch1.cdf();
    println!("  Median after merge: {}", merged_cdf.query(0.5));
}
