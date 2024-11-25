use std::fmt;

// used to print memory info

pub struct FmtBytes(pub u64);

impl fmt::Display for FmtBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const SUFFIX: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
        let mut idx = 0;
        let mut amount = self.0 as f64;
        loop {
            if amount < 1024.0 || idx == SUFFIX.len() - 1 {
                return write!(f, "{:.2} {}", amount, SUFFIX[idx]);
            }

            amount /= 1024.0;
            idx += 1;
        }
    }
}
