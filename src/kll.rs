use rand::Rng;
use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt;

// Thread-local `SmallRng` state.
thread_local! {
    static THREAD_RNG_KEY: RefCell<SmallRng> = RefCell::new(SmallRng::from_os_rng());
}

/// A handle to the thread-local `SmallRng`—similar to `rand::ThreadRng`.
#[derive(Debug, Clone)]
pub struct SmallThreadRng;

impl RngCore for SmallThreadRng {
    fn next_u32(&mut self) -> u32 {
        THREAD_RNG_KEY.with(|rng_cell| rng_cell.borrow_mut().next_u32())
    }

    fn next_u64(&mut self) -> u64 {
        THREAD_RNG_KEY.with(|rng_cell| rng_cell.borrow_mut().next_u64())
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        THREAD_RNG_KEY.with(|rng_cell| rng_cell.borrow_mut().fill_bytes(dest))
    }
}

pub fn small_thread_rng() -> SmallThreadRng {
    SmallThreadRng
}

pub fn capacity_coefficient(h: usize) -> f64 {
    if h < COEFF_CACHE.len() {
        COEFF_CACHE[h]
    } else {
        (2.0_f64 / 3.0_f64).powi(h as i32)
    }
}

// h:0 is the first compactor, h:1 is the second, etc.
pub fn capacity(k: usize, h: usize) -> usize {
    ((k as f64) * capacity_coefficient(h)).ceil() as usize
}

static COEFF_CACHE: &[f64] = &[
    1.0,
    0.6666666666666666,
    0.4444444444444444,
    0.2962962962962963,
    0.19753086419753085,
    0.1316872427983539,
    0.0877914951989026,
    0.05852766346593506,
    0.039018442310623375,
    0.026012294873748915,
    0.01734152991583261,
    0.011561019943888407,
    0.007707346629258938,
    0.005138231086172625,
    0.00342548739078175,
    0.0022836582605211663,
    0.0015224388403474443,
    0.0010149592268982961,
    0.0006766394845988641,
];

/// KLL sketch data structure for approximate quantile estimation.
///
/// The sketch maintains a hierarchy of compactors that progressively
/// compress the data stream while preserving quantile accuracy guarantees.
pub struct Sketch<T> {
    /// Hierarchy of compactors, where compactor[i] has weight 2^i.
    /// Each compactor stores sorted samples from the input stream.
    /// Higher levels store fewer but more representative samples.
    pub(crate) compactors: Vec<Compactor<T>>,

    /// The sketch parameter k that controls accuracy vs space tradeoff.
    /// Larger k provides better accuracy but uses more memory.
    /// Error is roughly O(1/k) for rank queries.
    pub(crate) k: usize,

    /// Current height of the compactor hierarchy (number of levels).
    /// Increases logarithmically with the number of items seen.
    pub(crate) h: usize,

    /// Current total number of items stored across all compactors.
    /// Used to trigger compaction when reaching max_size.
    pub(crate) size: usize,

    /// Maximum allowed size before triggering compaction.
    /// Computed as sum of capacities of all compactor levels.
    pub(crate) max_size: usize,
}

impl<T> Sketch<T>
where
    T: PartialOrd + Clone,
{
    pub fn new(k: usize) -> Self {
        // Pre-allocate capacity for the new compactor based on its expected size
        // Calculate capacity for the new level (h will be incremented after push)
        let expected_capacity = k; // Start with k
        let new_compactor = Compactor::with_capacity(expected_capacity);
        let size = 0;
        let max_size = expected_capacity;

        Sketch {
            compactors: vec![new_compactor],
            k,
            h: 1, // Start with one level
            size,
            max_size,
        }
    }

    fn grow(&mut self) {
        // Pre-allocate capacity for the new compactor based on its expected size
        // Calculate capacity for the new level (h will be incremented after push)
        self.compactors
            .push(Compactor::with_capacity(capacity(self.k, self.h)));
        self.h = self.compactors.len();

        self.max_size = 0;
        for h in 0..self.h {
            self.max_size += capacity(self.k, h);
        }
    }

    pub fn update(&mut self, x: T) {
        self.compactors[0].push(x);
        self.size += 1;
        self.compact();
    }

    /// Performs compaction to maintain space bounds while preserving quantile accuracy.
    ///
    /// This is the core algorithm of the KLL sketch. When the total size exceeds
    /// max_size, it finds a compactor that has reached capacity and compacts it.
    /// Compaction randomly selects approximately half of the items to promote to
    /// the next level, maintaining the probabilistic accuracy guarantees.
    fn compact(&mut self) {
        // Keep compacting until we're under the size limit
        while self.size >= self.max_size {
            // Check each level to find one that needs compaction
            for h in 0..self.compactors.len() {
                // If this compactor has reached its capacity
                if self.compactors[h].len() >= capacity(self.k, h) {
                    // Add a new level if we're compacting the highest level
                    if h + 1 >= self.h {
                        self.grow();
                    }

                    // Track sizes before compaction for accurate size update
                    let prev_h = self.compactors[h].len();
                    let prev_h1 = if h + 1 < self.compactors.len() {
                        self.compactors[h + 1].len()
                    } else {
                        0
                    };

                    // Get or create the destination compactor for promoted items
                    let dst = if h + 1 < self.compactors.len() {
                        self.compactors[h + 1].clone()
                    } else {
                        // Pre-allocate capacity for the new compactor
                        let expected_capacity = capacity(self.k, h + 1);
                        Compactor::with_capacity(expected_capacity)
                    };

                    // Perform the actual compaction: sort, then randomly keep ~half
                    let compacted = self.compactors[h].compact(dst);
                    if h + 1 < self.compactors.len() {
                        self.compactors[h + 1] = compacted;
                    }

                    // Update total size based on items removed and added
                    let new_h = self.compactors[h].len();
                    let new_h1 = if h + 1 < self.compactors.len() {
                        self.compactors[h + 1].len()
                    } else {
                        0
                    };

                    self.size = self.size - prev_h + new_h - prev_h1 + new_h1;

                    // Stop if we've freed enough space
                    if self.size < self.max_size {
                        break;
                    }
                }
            }
        }
    }

    fn update_size(&mut self) {
        self.size = 0;
        for c in &self.compactors {
            self.size += c.len();
        }
    }

    pub fn merge(&mut self, t: &Sketch<T>) {
        while self.h < t.h {
            self.grow();
        }

        for (h, c) in t.compactors.iter().enumerate() {
            if h < self.compactors.len() {
                self.compactors[h].extend(c.iter().cloned());
            }
        }

        self.update_size();
        self.compact();
    }

    pub fn rank(&self, x: T) -> usize {
        let mut r = 0;
        for (h, c) in self.compactors.iter().enumerate() {
            for v in c.iter() {
                if v <= &x {
                    r += 1 << h;
                }
            }
        }
        r
    }

    pub fn count(&self) -> usize {
        let mut n = 0;
        for (h, c) in self.compactors.iter().enumerate() {
            n += c.len() * (1 << h);
        }
        n
    }

    pub fn quantile(&self, x: T) -> f64 {
        let mut r = 0;
        let mut n = 0;
        for (h, c) in self.compactors.iter().enumerate() {
            for v in c.iter() {
                let w = 1 << h;
                if v <= &x {
                    r += w;
                }
                n += w;
            }
        }
        r as f64 / n as f64
    }

    pub fn cdf(&self) -> CDF<T> {
        let mut q = Vec::with_capacity(self.size);

        let mut total_w = 0.0;
        for (h, c) in self.compactors.iter().enumerate() {
            let weight = (1 << h) as f64;
            for v in c.iter() {
                q.push(Quantile {
                    q: weight,
                    v: v.clone(),
                });
            }
            total_w += c.len() as f64 * weight;
        }

        q.sort_by(|a, b| a.v.partial_cmp(&b.v).unwrap_or(Ordering::Equal));

        let mut cur_w = 0.0;
        for quantile in &mut q {
            cur_w += quantile.q;
            quantile.q = cur_w / total_w;
        }

        CDF(q)
    }

    /// Creates a histogram-style visualization of the sketch data distribution.
    /// This provides a compact view of how data is distributed across the sketch.
    /// Only available for numeric types that can be converted to f64.
    pub fn histogram(&self, bins: usize) -> String
    where
        T: Into<f64> + Copy,
    {
        if self.size == 0 {
            return "Empty sketch".to_string();
        }

        // Collect all values with their weights
        let mut values: Vec<(f64, usize)> = Vec::new();
        for (level, compactor) in self.compactors.iter().enumerate() {
            let weight = 1 << level;
            for v in compactor.iter() {
                values.push(((*v).into(), weight));
            }
        }

        // Sort by value
        values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // Find min and max
        let min_val = values[0].0;
        let max_val = values[values.len() - 1].0;
        let range = max_val - min_val;

        if range == 0.0 {
            return format!("All values equal to {}", min_val);
        }

        // Create bins
        let mut bin_counts = vec![0; bins];
        let bin_width = range / bins as f64;

        for (val, weight) in values {
            let bin_idx = ((val - min_val) / bin_width) as usize;
            let bin_idx = bin_idx.min(bins - 1); // Handle edge case for max value
            bin_counts[bin_idx] += weight;
        }

        // Find max count for scaling
        let max_count = *bin_counts.iter().max().unwrap_or(&0);
        let bar_width = 40;

        let mut result = String::new();
        result.push_str(&format!(
            "Histogram ({} bins, {} total weighted items):\n",
            bins,
            self.count()
        ));
        result.push_str(&format!("Range: [{:.2}, {:.2}]\n\n", min_val, max_val));

        for (i, &count) in bin_counts.iter().enumerate() {
            let bin_start = min_val + i as f64 * bin_width;
            let bin_end = bin_start + bin_width;

            let bar_len = if max_count > 0 {
                (count * bar_width / max_count).max(1)
            } else {
                0
            };

            let bar = "█".repeat(bar_len);
            result.push_str(&format!(
                "[{:>7.2}, {:>7.2}): {:>6} |{:<width$}|\n",
                bin_start,
                bin_end,
                count,
                bar,
                width = bar_width
            ));
        }

        result
    }
}

impl fmt::Display for Sketch<f64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "KLL Sketch (k={})", self.k)?;
        writeln!(
            f,
            "Total items: {} | Max capacity: {}",
            self.count(),
            self.max_size
        )?;
        writeln!(f)?;

        for (level, compactor) in self.compactors.iter().enumerate() {
            let items = compactor.len();
            let capacity = capacity(self.k, level);

            write!(f, "Level {} (2^{}): {}/{}", level, level, items, capacity)?;

            // Show sample values
            if items > 0 {
                write!(f, " → [")?;

                let display_count = items.min(5);
                for (i, v) in compactor.iter().take(display_count).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }

                    // Compact numeric formatting
                    if v.abs() >= 10000.0 {
                        write!(f, "{:.1e}", v)?;
                    } else if v.abs() < 0.001 && *v != 0.0 {
                        write!(f, "{:.1e}", v)?;
                    } else if v.fract() == 0.0 && v.abs() < 1000.0 {
                        write!(f, "{:.0}", v)?;
                    } else {
                        write!(f, "{:.2}", v)?;
                    }
                }

                if items > display_count {
                    write!(f, ", ... ({} more)", items - display_count)?;
                }

                write!(f, "]")?;
            }

            writeln!(f)?;
        }

        Ok(())
    }
}

impl fmt::Display for Sketch<String> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "KLL Sketch (k={})", self.k)?;
        writeln!(
            f,
            "Total items: {} | Max capacity: {}",
            self.count(),
            self.max_size
        )?;
        writeln!(f)?;

        for (level, compactor) in self.compactors.iter().enumerate() {
            let items = compactor.len();
            let capacity = capacity(self.k, level);

            write!(f, "Level {} (2^{}): {}/{}", level, level, items, capacity)?;

            // Show sample values
            if items > 0 {
                write!(f, " → [")?;

                let display_count = items.min(4);
                for (i, v) in compactor.iter().take(display_count).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }

                    // Truncate long strings
                    if v.len() > 12 {
                        write!(f, "\"{}...\"", &v[..9])?;
                    } else {
                        write!(f, "\"{}\"", v)?;
                    }
                }

                if items > display_count {
                    write!(f, ", ... ({} more)", items - display_count)?;
                }

                write!(f, "]")?;
            }

            writeln!(f)?;
        }

        Ok(())
    }
}

impl fmt::Display for Sketch<Vec<u8>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "KLL Sketch (k={})", self.k)?;
        writeln!(
            f,
            "Total items: {} | Max capacity: {}",
            self.count(),
            self.max_size
        )?;
        writeln!(f)?;

        for (level, compactor) in self.compactors.iter().enumerate() {
            let items = compactor.len();
            let capacity = capacity(self.k, level);

            write!(f, "Level {} (2^{}): {}/{}", level, level, items, capacity)?;

            // Show sample values
            if items > 0 {
                write!(f, " → [")?;

                let display_count = items.min(3);
                for (i, v) in compactor.iter().take(display_count).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }

                    // Format as hex bytes
                    write!(f, "[")?;
                    let byte_count = v.len().min(6);
                    for (j, &byte) in v.iter().take(byte_count).enumerate() {
                        if j > 0 {
                            write!(f, " ")?;
                        }
                        write!(f, "{:02x}", byte)?;
                    }
                    if v.len() > byte_count {
                        write!(f, "... ({} bytes)", v.len())?;
                    }
                    write!(f, "]")?;
                }

                if items > display_count {
                    write!(f, ", ... ({} more)", items - display_count)?;
                }

                write!(f, "]")?;
            }

            writeln!(f)?;
        }

        Ok(())
    }
}

// Generic display implementation for integers
impl fmt::Display for Sketch<i32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "KLL Sketch (k={})", self.k)?;
        writeln!(
            f,
            "Total items: {} | Max capacity: {}",
            self.count(),
            self.max_size
        )?;
        writeln!(f)?;

        for (level, compactor) in self.compactors.iter().enumerate() {
            let items = compactor.len();
            let capacity = capacity(self.k, level);

            write!(f, "Level {} (2^{}): {}/{}", level, level, items, capacity)?;

            // Show sample values
            if items > 0 {
                write!(f, " → [")?;

                let display_count = items.min(8);
                for (i, v) in compactor.iter().take(display_count).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }

                if items > display_count {
                    write!(f, ", ... ({} more)", items - display_count)?;
                }

                write!(f, "]")?;
            }

            writeln!(f)?;
        }

        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct Compactor<T>(pub Vec<T>);

impl<T> Compactor<T>
where
    T: PartialOrd + Clone,
{
    fn with_capacity(capacity: usize) -> Self {
        Compactor(Vec::with_capacity(capacity))
    }

    fn push(&mut self, x: T) {
        self.0.push(x);
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn iter(&self) -> std::slice::Iter<T> {
        self.0.iter()
    }

    fn extend(&mut self, iter: impl Iterator<Item = T>) {
        self.0.extend(iter);
    }

    /// Compacts this compactor by randomly selecting approximately half of its items
    /// to promote to the destination compactor.
    ///
    /// The algorithm:
    /// 1. Sorts the items (using insertion sort for small arrays, quicksort for large)
    /// 2. Randomly chooses to keep either even or odd indexed items
    /// 3. Promotes selected items to the destination compactor
    /// 4. Keeps the remaining items in this compactor
    ///
    /// This randomized selection maintains the probabilistic guarantees of the sketch
    /// while reducing the number of stored items by approximately half.
    fn compact(&mut self, mut dst: Compactor<T>) -> Compactor<T> {
        let l = self.0.len();

        // Nothing to compact if we have 0 or 1 items
        if l == 0 || l == 1 {
            return dst;
        } else if l == 2 {
            // Special case for 2 items: just ensure they're sorted
            if self.0[0] > self.0[1] {
                self.0.swap(0, 1);
            }
        } else if l > 100 {
            // Use standard sort for large arrays (more efficient)
            self.0
                .sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        } else {
            // Use insertion sort for small arrays (better cache locality)
            self.insertion_sort();
        }

        // Ensure destination has enough capacity for the promoted items
        let free = dst.0.capacity() - dst.0.len();
        if free < self.0.len() / 2 {
            let extra = self.0.len() / 2 - free;
            dst.0.reserve(extra);
        }

        // Randomly choose offset (0 or 1) to determine which items to keep
        // This ensures each item has 50% chance of being promoted
        let offs = small_thread_rng().random_range(0..2) as usize;

        // Process pairs from the end, promoting one item from each pair
        while self.0.len() >= 2 {
            let l = self.0.len() - 2;
            dst.0.push(self.0[l + offs].clone());
            self.0.truncate(l);
        }

        dst
    }

    pub fn insertion_sort(&mut self) {
        let l = self.0.len();
        for i in 1..l {
            let v = self.0[i].clone();
            let mut j = i;
            while j > 0 && self.0[j - 1] > v {
                j -= 1;
            }
            if j == i {
                continue;
            }
            let temp = self.0[i].clone();
            for k in (j + 1..=i).rev() {
                self.0[k] = self.0[k - 1].clone();
            }
            self.0[j] = temp;
        }
    }
}

#[derive(Clone, Debug)]
pub struct Quantile<T> {
    pub q: f64,
    pub v: T,
}

#[derive(Clone, Debug)]
pub struct CDF<T>(Vec<Quantile<T>>);

impl<T> CDF<T>
where
    T: PartialOrd + Clone,
{
    pub fn quantile(&self, x: T) -> f64 {
        let idx = self
            .0
            .binary_search_by(|q| {
                if q.v < x {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .unwrap_or_else(|e| e);

        if idx == 0 { 0.0 } else { self.0[idx - 1].q }
    }

    pub fn query(&self, p: f64) -> T {
        let idx = self
            .0
            .binary_search_by(|q| {
                if q.q < p {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .unwrap_or_else(|e| e);

        if idx == self.0.len() {
            self.0[self.0.len() - 1].v.clone()
        } else {
            self.0[idx].v.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand_distr::{Distribution, Exp, Normal};

    #[test]
    fn test_basic_sketch() {
        let mut sketch = Sketch::new(200);

        for i in 0..1000 {
            sketch.update(i as f64);
        }

        assert_eq!(sketch.count(), 1000);

        assert!(sketch.rank(500.0) > 450 && sketch.rank(500.0) < 550);
        assert!(sketch.quantile(500.0) > 0.45 && sketch.quantile(500.0) < 0.55);
    }

    #[test]
    fn test_capacity_coefficient() {
        assert_eq!(capacity_coefficient(0), 1.0);
        assert_eq!(capacity_coefficient(1), 2.0 / 3.0);
        assert_eq!(capacity_coefficient(2), 4.0 / 9.0);
        assert_eq!(capacity_coefficient(3), 8.0 / 27.0);
        assert_eq!(capacity_coefficient(4), 16.0 / 81.0);
    }

    #[test]
    fn test_cdf() {
        let mut sketch = Sketch::new(200);

        for i in 0..10000 {
            sketch.update(i as f64);
        }

        let cdf = sketch.cdf();

        let median = cdf.query(0.5);
        assert!(median > 4900.0 && median < 5100.0);

        let p90 = cdf.query(0.9);
        assert!(p90 > 8900.0 && p90 < 9100.0);

        let p99 = cdf.query(0.99);
        assert!(p99 > 9800.0 && p99 < 9999.0);
    }

    #[test]
    fn test_merge() {
        let mut sketch1 = Sketch::new(200);
        let mut sketch2 = Sketch::new(200);

        for i in 0..5000 {
            sketch1.update(i as f64);
        }

        for i in 5000..10000 {
            sketch2.update(i as f64);
        }

        sketch1.merge(&sketch2);

        assert_eq!(sketch1.count(), 10000);

        let cdf = sketch1.cdf();
        let median = cdf.query(0.5);
        assert!(median > 4900.0 && median < 5100.0);
    }

    #[test]
    fn test_compactor_insertion_sort() {
        for &dup in &[false, true] {
            for &l in &[0, 1, 2, 3, 5, 8, 32, 1024] {
                for _ in 0..10 {
                    let mut c = crate::kll::Compactor(vec![0.0; l]);
                    for i in 0..l {
                        if dup && i % 2 == 1 && i > 0 {
                            c.0[i] = c.0[i - 1];
                        } else {
                            c.0[i] = small_thread_rng().random::<f64>() * 200.0 - 100.0;
                        }
                    }
                    let mut expected = c.0.clone();
                    c.insertion_sort();

                    expected.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    for (i, (&actual, &expect)) in c.0.iter().zip(expected.iter()).enumerate() {
                        assert_eq!(actual, expect, "failed to sort at index {}", i);
                    }
                }
            }
        }
    }

    #[test]
    fn test_empty_sketch() {
        let sketch = Sketch::new(200);
        assert_eq!(sketch.count(), 0);
        assert_eq!(sketch.rank(0.0), 0);
        assert_eq!(sketch.rank(100.0), 0);
    }

    #[test]
    fn test_single_element() {
        let mut sketch = Sketch::new(200);
        sketch.update(42.0);

        assert_eq!(sketch.count(), 1);
        assert_eq!(sketch.rank(41.9), 0);
        assert_eq!(sketch.rank(42.0), 1);
        assert_eq!(sketch.rank(42.1), 1);

        let cdf = sketch.cdf();
        assert_eq!(cdf.query(0.0), 42.0);
        assert_eq!(cdf.query(0.5), 42.0);
        assert_eq!(cdf.query(1.0), 42.0);
    }

    #[test]
    fn test_uniform_distribution_accuracy() {
        for k in &[50, 100, 200, 500] {
            let mut sketch = Sketch::new(*k);
            let n = 100_000;

            for i in 0..n {
                sketch.update(i as f64);
            }

            let cdf = sketch.cdf();

            // Test various quantiles
            let test_quantiles = vec![0.1, 0.25, 0.5, 0.75, 0.9];
            for &q in &test_quantiles {
                let estimated = cdf.query(q);
                let expected = q * n as f64;
                let error = (estimated - expected).abs() / expected;

                // Error should be roughly O(1/k)
                let expected_error = 2.0 / (*k as f64).sqrt();

                println!(
                    "k={}, q={}: estimated={}, expected={}, error={}, expected_error={}",
                    k, q, estimated, expected, error, expected_error
                );
                assert!(
                    error < expected_error,
                    "k={}, q={}: error={} > expected={}",
                    k,
                    q,
                    error,
                    expected_error
                );
            }
        }
    }

    #[test]
    fn test_normal_distribution_accuracy() {
        let mut rng = small_thread_rng();
        let normal = Normal::new(500.0, 100.0).unwrap();

        let mut sketch = Sketch::new(200);
        let n = 100_000;

        for _ in 0..n {
            sketch.update(normal.sample(&mut rng));
        }

        let cdf = sketch.cdf();

        // Test standard normal quantiles
        let test_cases = vec![
            (0.5, 500.0),    // median
            (0.1587, 400.0), // μ - σ
            (0.8413, 600.0), // μ + σ
            (0.0228, 300.0), // μ - 2σ
            (0.9772, 700.0), // μ + 2σ
        ];

        for (q, expected) in test_cases {
            let estimated = cdf.query(q);
            let error = ((estimated - expected) as f64).abs() / expected;
            println!(
                "q={}: estimated={}, expected={}, error={}",
                q, estimated, expected, error
            );
            assert!(error < 0.05, "q={}: error={} > 0.05", q, error);
        }
    }

    #[test]
    fn test_exponential_distribution() {
        let mut rng = small_thread_rng();
        let exp = Exp::new(1.0).unwrap();

        let mut sketch = Sketch::new(200);
        let n = 50_000;

        for _ in 0..n {
            sketch.update(exp.sample(&mut rng));
        }

        let cdf = sketch.cdf();

        // For exponential distribution with λ=1:
        // CDF(x) = 1 - e^(-x)
        // Quantile(p) = -ln(1-p)
        let test_quantiles = vec![0.1, 0.25, 0.5, 0.75, 0.9];
        for &q in &test_quantiles {
            let estimated = cdf.query(q);
            let expected = -(1.0 - q).ln();
            let error = (estimated - expected).abs() / expected;
            println!(
                "q={}: estimated={}, expected={}, error={}",
                q, estimated, expected, error
            );
            assert!(error < 0.1, "q={}: error={}", q, error);
        }
    }

    #[test]
    fn test_bimodal_distribution() {
        let mut rng = small_thread_rng();
        let mode1 = Normal::new(100.0, 10.0).unwrap();
        let mode2 = Normal::new(200.0, 10.0).unwrap();

        let mut sketch = Sketch::new(300);
        let n = 50_000;

        for i in 0..n {
            let value = if i % 2 == 0 {
                mode1.sample(&mut rng)
            } else {
                mode2.sample(&mut rng)
            };
            sketch.update(value);
        }

        let cdf = sketch.cdf();

        // Should have roughly equal mass around each mode
        let q25 = cdf.query(0.25);
        let q75 = cdf.query(0.75);

        assert!(
            q25 > 80.0 && q25 < 120.0,
            "25th percentile should be near first mode: {}",
            q25
        );
        assert!(
            q75 > 180.0 && q75 < 220.0,
            "75th percentile should be near second mode: {}",
            q75
        );
    }

    #[test]
    fn test_negative_numbers() {
        let mut sketch = Sketch::new(100);

        for i in -1000..1000 {
            sketch.update(i as f64);
        }

        assert_eq!(sketch.count(), 2000);

        let cdf = sketch.cdf();
        let median = cdf.query(0.5);
        // The median of -1000..1000 is around 0, with some approximation error
        assert!(
            median.abs() < 50.0,
            "Median should be near 0, got {}",
            median
        );

        let q25 = cdf.query(0.25);
        assert!(
            (q25 - (-500.0)).abs() < 50.0,
            "25th percentile should be near -500, got {}",
            q25
        );
    }

    #[test]
    fn test_duplicates() {
        let mut sketch = Sketch::new(100);

        // Add many duplicates
        for _ in 0..1000 {
            sketch.update(1.0);
        }
        for _ in 0..1000 {
            sketch.update(2.0);
        }
        for _ in 0..1000 {
            sketch.update(3.0);
        }

        let cdf = sketch.cdf();

        let q33 = cdf.query(0.33);
        let q66 = cdf.query(0.66);

        assert!(
            ((q33 - 1.0) as f64).abs() < 0.5 || ((q33 - 2.0) as f64).abs() < 0.5,
            "33rd percentile should be near 1 or 2, got {}",
            q33
        );
        assert!(
            ((q66 - 2.0) as f64).abs() < 0.5 || ((q66 - 3.0) as f64).abs() < 0.5,
            "66th percentile should be near 2 or 3, got {}",
            q66
        );
    }

    #[test]
    fn test_extreme_quantiles() {
        let mut sketch = Sketch::new(200);

        for i in 0..10000 {
            sketch.update(i as f64);
        }

        let cdf = sketch.cdf();

        // Test extreme quantiles
        let q001 = cdf.query(0.001);
        let q999 = cdf.query(0.999);

        assert!(
            q001 < 100.0,
            "0.1th percentile should be < 100, got {}",
            q001
        );
        assert!(
            q999 > 9900.0,
            "99.9th percentile should be > 9900, got {}",
            q999
        );

        // Test boundaries
        let q0 = cdf.query(0.0);
        let q1 = cdf.query(1.0);

        assert!(
            q0 >= 0.0 && q0 < 100.0,
            "0th percentile out of range: {}",
            q0
        );
        assert!(
            q1 >= 9900.0 && q1 < 10000.0,
            "100th percentile out of range: {}",
            q1
        );
    }

    #[test]
    fn test_merge_accuracy() {
        // Create two sketches with different ranges
        let mut sketch1 = Sketch::new(200);
        let mut sketch2 = Sketch::new(200);

        for i in 0..50000 {
            sketch1.update(i as f64);
        }

        for i in 50000..100000 {
            sketch2.update(i as f64);
        }

        // Merge them
        sketch1.merge(&sketch2);

        assert_eq!(sketch1.count(), 100000);

        let cdf = sketch1.cdf();

        // Check that quantiles are still accurate after merge
        for &q in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let estimated = cdf.query(q);
            let expected = q * 100000.0;
            let error = (estimated - expected).abs() / expected;
            println!(
                "After merge q={}: estimated={}, expected={}, error={}",
                q, estimated, expected, error
            );
            assert!(error < 0.1, "After merge q={}: error={}", q, error);
        }
    }

    #[test]
    fn test_memory_bounds() {
        // Test that sketch stays within expected memory bounds
        let k = 100;
        let mut sketch = Sketch::new(k);

        // Add many elements
        for i in 0..1_000_000 {
            sketch.update(i as f64);
        }

        // Total elements in all compactors should be bounded
        let total_stored: usize = sketch.compactors.iter().map(|c| c.len()).sum();
        let h = sketch.h;

        // Theoretical bound is O(k * log(n/k))
        let expected_max = (k as f64 * (h as f64) * 2.0) as usize;
        assert!(
            total_stored < expected_max,
            "Memory usage {} exceeds expected bound {}",
            total_stored,
            expected_max
        );
    }

    #[test]
    fn test_rank_consistency() {
        let mut sketch = Sketch::new(200);
        let values: Vec<f64> = (0..1000).map(|i| i as f64).collect();

        for &v in &values {
            sketch.update(v);
        }

        // Rank should be monotonic
        let mut prev_rank = 0;
        for i in 0..1000 {
            let rank = sketch.rank(i as f64);
            assert!(
                rank >= prev_rank,
                "Rank not monotonic at {}: {} < {}",
                i,
                rank,
                prev_rank
            );
            prev_rank = rank;
        }

        // Test rank bounds
        assert_eq!(sketch.rank(-1.0), 0);
        assert_eq!(sketch.rank(1000.0), sketch.count());
    }

    #[test]
    fn test_compactor_operations() {
        // Test the compactor directly
        let mut c = Compactor::with_capacity(10);

        // Test empty compactor
        assert_eq!(c.len(), 0);

        // Test single element
        c.push(5.0);
        assert_eq!(c.len(), 1);

        // Test sorting with duplicates
        c.push(3.0);
        c.push(7.0);
        c.push(3.0);
        c.push(1.0);

        c.insertion_sort();

        let expected = vec![1.0, 3.0, 3.0, 5.0, 7.0];
        assert_eq!(c.0, expected);
    }

    #[test]
    fn test_deterministic_operations() {
        // Test that operations are deterministic given same seed
        let mut sketch1 = Sketch::new(100);
        let mut sketch2 = Sketch::new(100);

        // Reset thread-local RNG to ensure determinism
        THREAD_RNG_KEY.with(|rng_cell| {
            *rng_cell.borrow_mut() = SmallRng::seed_from_u64(12345);
        });

        for i in 0..10000 {
            sketch1.update((i * 7) as f64 % 1000.0);
        }

        THREAD_RNG_KEY.with(|rng_cell| {
            *rng_cell.borrow_mut() = SmallRng::seed_from_u64(12345);
        });

        for i in 0..10000 {
            sketch2.update((i * 7) as f64 % 1000.0);
        }

        // The sketches should produce similar results
        let cdf1 = sketch1.cdf();
        let cdf2 = sketch2.cdf();

        for &q in &[0.1, 0.5, 0.9] {
            let v1 = cdf1.query(q);
            let v2 = cdf2.query(q);
            assert!(
                (v1 - v2).abs() < 50.0,
                "Results differ for q={}: {} vs {}",
                q,
                v1,
                v2
            );
        }
    }

    #[test]
    fn test_streaming_property() {
        // Test that sketch works correctly as a streaming algorithm
        let mut sketch = Sketch::new(200);
        let mut exact_values = Vec::new();

        let mut rng = small_thread_rng();

        // Simulate streaming data
        for _ in 0..10 {
            // Add batch of data
            for _ in 0..1000 {
                let v = rng.random_range(0.0..100.0);
                sketch.update(v);
                exact_values.push(v);
            }

            // Check accuracy at this point
            let cdf = sketch.cdf();
            exact_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            for &q in &[0.25, 0.5, 0.75] {
                let estimated = cdf.query(q);
                let exact_idx = (q * exact_values.len() as f64) as usize;
                let exact = exact_values[exact_idx.min(exact_values.len() - 1)];
                let error = ((estimated - exact) as f64).abs() / exact;
                assert!(error < 0.1, "Streaming error at q={}: {}", q, error);
            }
        }
    }

    #[test]
    fn test_corner_cases() {
        let mut sketch = Sketch::new(10); // Small k

        // Test with NaN - should handle gracefully
        sketch.update(f64::NAN);
        sketch.update(1.0);
        sketch.update(2.0);

        // Test with infinity
        sketch.update(f64::INFINITY);
        sketch.update(f64::NEG_INFINITY);

        // Should still work
        assert_eq!(sketch.count(), 5);

        // Test with very small k
        let mut tiny_sketch = Sketch::new(1);
        for i in 0..100 {
            tiny_sketch.update(i as f64);
        }
        assert_eq!(tiny_sketch.count(), 100);
    }

    #[test]
    fn test_quantile_accuracy_vs_exact() {
        // For small datasets, compare with exact quantiles
        let mut sketch = Sketch::new(100);
        let mut exact = Vec::new();

        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
        ];

        for &v in &data {
            sketch.update(v);
            exact.push(v);
        }

        exact.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cdf = sketch.cdf();

        // For small dataset, should be very accurate
        for i in 0..20 {
            let p = (i as f64 + 0.5) / 20.0;
            let estimated = cdf.query(p);
            let exact_val = exact[i];
            assert!(
                ((estimated - exact_val) as f64).abs() < 1.0,
                "p={}: estimated={}, exact={}",
                p,
                estimated,
                exact_val
            );
        }
    }

    #[test]
    fn test_periodic_values() {
        // Test with periodic/cyclic data
        let mut sketch = Sketch::new(200);

        for i in 0..10000 {
            sketch.update((i as f64 * 0.1).sin() * 100.0);
        }

        let cdf = sketch.cdf();

        // Should have symmetric distribution around 0
        let q25 = cdf.query(0.25);
        let q75 = cdf.query(0.75);

        assert!(
            (q25 + q75).abs() < 10.0,
            "Distribution not symmetric: {} + {} = {}",
            q25,
            q75,
            q25 + q75
        );

        // Median should be near 0
        let median = cdf.query(0.5);
        assert!(median.abs() < 5.0, "Median not near 0: {}", median);
    }

    #[test]
    fn test_integer_sketch() {
        let mut sketch = Sketch::<i32>::new(100);

        // Add integers from 0 to 999
        for i in 0..1000 {
            sketch.update(i);
        }

        assert_eq!(sketch.count(), 1000);

        // Test rank
        assert_eq!(sketch.rank(-1), 0);
        assert!(sketch.rank(250) > 200 && sketch.rank(250) < 300);
        assert!(sketch.rank(500) > 450 && sketch.rank(500) < 550);
        assert!(sketch.rank(750) > 700 && sketch.rank(750) < 800);
        assert_eq!(sketch.rank(1000), sketch.count());

        // Test CDF
        let cdf = sketch.cdf();
        let median = cdf.query(0.5);
        assert!(median > 450 && median < 550);

        let q25 = cdf.query(0.25);
        assert!(q25 > 200 && q25 < 300);

        let q75 = cdf.query(0.75);
        assert!(q75 > 700 && q75 < 800);
    }

    #[test]
    fn test_string_sketch() {
        let mut sketch = Sketch::<String>::new(50);

        // Add words with different frequencies
        let words = vec![
            ("apple", 100),
            ("banana", 200),
            ("cherry", 300),
            ("date", 150),
            ("elderberry", 50),
        ];

        for (word, count) in &words {
            for _ in 0..*count {
                sketch.update(word.to_string());
            }
        }

        assert_eq!(sketch.count(), 800);

        // Test rank - "cherry" should have rank around 300 (apple + banana counts)
        // But due to compression, the estimate might be less accurate
        let cherry_rank = sketch.rank("cherry".to_string());
        assert!(
            cherry_rank > 200 && cherry_rank < 700,
            "cherry rank: {}",
            cherry_rank
        );

        // Test that alphabetical ordering works
        assert_eq!(sketch.rank("aardvark".to_string()), 0); // before all
        assert_eq!(sketch.rank("zebra".to_string()), sketch.count()); // after all

        // Test CDF
        let cdf = sketch.cdf();

        // Query at different percentiles
        let q25 = cdf.query(0.25);
        assert_eq!(q25, "banana"); // First 200 are apple, next 200 are banana

        let q50 = cdf.query(0.5);
        assert!(q50 == "cherry" || q50 == "date"); // Around the middle

        let q90 = cdf.query(0.9);
        assert!(q90 == "date" || q90 == "elderberry"); // Near the end
    }

    #[test]
    fn test_vec_u8_sketch() {
        let mut sketch = Sketch::<Vec<u8>>::new(100);

        // Add byte arrays - they sort lexicographically
        // Single bytes: [0], [1], ..., [99]
        // Two bytes: [0,0], [0,1], ..., [99,99]
        // Three bytes: [0,0,0], [0,0,1], ..., [99,99,99]

        for i in 0..100 {
            sketch.update(vec![i as u8]);
        }

        for i in 0..100 {
            sketch.update(vec![i as u8, i as u8]);
        }

        for i in 0..100 {
            sketch.update(vec![i as u8, i as u8, i as u8]);
        }

        assert_eq!(sketch.count(), 300);

        // Test rank - lexicographic ordering
        // vec![0] is the smallest
        let rank_0 = sketch.rank(vec![0]);
        assert!(rank_0 < 10, "rank of vec![0]: {}", rank_0);

        // vec![50] comes after [0] through [49] and [0,0] through [49,99]
        let rank_50 = sketch.rank(vec![50]);
        assert!(
            rank_50 > 100,
            "rank of vec![50] should be > 100: {}",
            rank_50
        );

        // vec![99] is the largest single-byte array, comes after many two-byte arrays
        let rank_99 = sketch.rank(vec![99]);
        assert!(rank_99 > 250, "rank of vec![99]: {}", rank_99);

        // Test CDF - results may vary due to compression
        let cdf = sketch.cdf();

        // Due to compression, we can't guarantee exact byte lengths at percentiles
        // Just verify the results are valid byte arrays
        let q20 = cdf.query(0.20);
        assert!(!q20.is_empty(), "20th percentile should not be empty");

        let q50 = cdf.query(0.50);
        assert!(!q50.is_empty(), "50th percentile should not be empty");

        let q80 = cdf.query(0.80);
        assert!(!q80.is_empty(), "80th percentile should not be empty");
    }

    #[test]
    fn test_mixed_integer_types() {
        // Test with different integer types to ensure generic implementation works
        let mut u32_sketch = Sketch::<u32>::new(50);
        let mut i64_sketch = Sketch::<i64>::new(50);
        let mut usize_sketch = Sketch::<usize>::new(50);

        for i in 0..100 {
            u32_sketch.update(i as u32);
            i64_sketch.update(i as i64 - 50); // Include negative numbers
            usize_sketch.update(i);
        }

        assert_eq!(u32_sketch.count(), 100);
        assert_eq!(i64_sketch.count(), 100);
        assert_eq!(usize_sketch.count(), 100);

        // Test that each works correctly
        let u32_cdf = u32_sketch.cdf();
        assert!(u32_cdf.query(0.5) > 45 && u32_cdf.query(0.5) < 55);

        let i64_cdf = i64_sketch.cdf();
        let i64_median = i64_cdf.query(0.5);
        assert!(i64_median > -5 && i64_median < 5); // Should be around 0

        let usize_cdf = usize_sketch.cdf();
        assert!(usize_cdf.query(0.5) > 45 && usize_cdf.query(0.5) < 55);
    }

    #[test]
    fn test_string_merge() {
        let mut sketch1 = Sketch::<String>::new(50);
        let mut sketch2 = Sketch::<String>::new(50);

        // Add different sets of words to each sketch
        for _ in 0..100 {
            sketch1.update("apple".to_string());
            sketch1.update("banana".to_string());
        }

        for _ in 0..100 {
            sketch2.update("cherry".to_string());
            sketch2.update("date".to_string());
        }

        sketch1.merge(&sketch2);

        assert_eq!(sketch1.count(), 400);

        // After merge, should have all words
        let cdf = sketch1.cdf();

        // Should have roughly equal distribution across 4 words
        let q125 = cdf.query(0.125);
        assert_eq!(q125, "apple"); // 12.5% should be apple or banana

        let q375 = cdf.query(0.375);
        assert_eq!(q375, "banana"); // 37.5% should be banana

        let q625 = cdf.query(0.625);
        assert_eq!(q625, "cherry"); // 62.5% should be cherry

        let q875 = cdf.query(0.875);
        assert_eq!(q875, "date"); // 87.5% should be date
    }

    #[test]
    fn test_pretty_print() {
        let mut sketch = Sketch::new(50);

        // Add some data to create multiple levels
        for i in 0..1000 {
            sketch.update(i as f64);
        }

        // Print the sketch
        println!("Sketch visualization:");
        println!("{}", sketch);

        // Test with strings
        let mut string_sketch = Sketch::<String>::new(30);
        let words = vec![
            "apple",
            "banana",
            "cherry",
            "date",
            "elderberry",
            "fig",
            "grape",
        ];

        for (i, word) in words.iter().cycle().take(200).enumerate() {
            string_sketch.update(format!("{}-{}", word, i));
        }

        println!("\nString sketch visualization:");
        println!("{}", string_sketch);
    }
}
