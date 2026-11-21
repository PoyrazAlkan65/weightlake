use std::ops::Range;

/// Target chunk size. Chunks are cut on row boundaries so that a chunk is a
/// contiguous block of whole rows of the matrix whenever a row fits.
pub const DEFAULT_CHUNK_BYTES: usize = 4 << 20;

/// Split a tensor's `total` bytes into chunk ranges.
///
/// - 2D+ tensors: cut on row boundaries (row = product of trailing dims).
/// - Rows wider than `target` (or 1D tensors): cut on element boundaries.
pub fn split_ranges(total: usize, shape: &[u64], elem: usize, target: usize) -> Vec<Range<usize>> {
    if total == 0 {
        return Vec::new();
    }
    let elem = elem.max(1);
    let row_bytes = if shape.len() >= 2 {
        shape[1..].iter().product::<u64>() as usize * elem
    } else {
        elem
    };
    let unit = if row_bytes > 0 && row_bytes <= target {
        row_bytes * (target / row_bytes).max(1)
    } else {
        ((target / elem) * elem).max(elem)
    };
    let mut out = Vec::with_capacity(total.div_ceil(unit));
    let mut start = 0;
    while start < total {
        let end = (start + unit).min(total);
        out.push(start..end);
        start = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_not_split() {
        // 1000 rows x 1000 cols f32 = 4000 bytes per row
        let r = split_ranges(4_000_000, &[1000, 1000], 4, 1 << 20);
        assert!(r.iter().all(|x| x.len() % 4000 == 0));
        assert_eq!(r.last().unwrap().end, 4_000_000);
        assert_eq!(r.first().unwrap().len(), 4000 * 262);
    }

    #[test]
    fn wide_rows_split_on_elements() {
        let r = split_ranges(10_000, &[1, 2500], 4, 1000);
        assert!(r.iter().all(|x| x.len() % 4 == 0));
        assert_eq!(r.len(), 10);
    }

    #[test]
    fn empty() {
        assert!(split_ranges(0, &[0, 10], 4, 100).is_empty());
    }
}
