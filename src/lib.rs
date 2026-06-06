//! # ternary-haar
//!
//! Haar wavelet transform for ternary-valued signals {-1, 0, +1}.
//!
//! This crate implements the Haar wavelet decomposition and reconstruction,
//! multi-resolution analysis, wavelet coefficient manipulation, and compression
//! via coefficient thresholding. The Haar wavelet is the simplest wavelet, making
//! it ideal for piecewise-constant ternary signals.

/// A single level of Haar wavelet decomposition, containing approximation and detail coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct HaarLevel {
    /// Approximation coefficients (low-pass / scaling function output)
    pub approximation: Vec<f64>,
    /// Detail coefficients (high-pass / wavelet function output)
    pub detail: Vec<f64>,
}

/// Full multi-resolution Haar decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct HaarDecomposition {
    /// Decomposition levels, from finest to coarsest.
    /// levels[0] is the first decomposition of the original signal.
    /// The last level contains the coarsest approximation and detail.
    pub levels: Vec<HaarLevel>,
    /// The original signal length (before decomposition).
    pub original_len: usize,
}

/// Perform a single level of the Haar wavelet transform on a signal.
///
/// For a signal of length N, produces approximation coefficients of length N/2
/// and detail coefficients of length N/2.
///
/// Approximation: a[k] = (x[2k] + x[2k+1]) / sqrt(2)
/// Detail:        d[k] = (x[2k] - x[2k+1]) / sqrt(2)
fn haar_single_level(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    assert!(n >= 2 && n % 2 == 0, "signal length must be even and >= 2");
    let half = n / 2;
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

    let mut approx = vec![0.0; half];
    let mut detail = vec![0.0; half];

    for k in 0..half {
        approx[k] = (signal[2 * k] + signal[2 * k + 1]) * inv_sqrt2;
        detail[k] = (signal[2 * k] - signal[2 * k + 1]) * inv_sqrt2;
    }

    (approx, detail)
}

/// Reconstruct a signal from one level of Haar approximation and detail coefficients.
///
/// x[2k]   = (a[k] + d[k]) / sqrt(2)
/// x[2k+1] = (a[k] - d[k]) / sqrt(2)
fn inverse_haar_single_level(approx: &[f64], detail: &[f64]) -> Vec<f64> {
    assert_eq!(approx.len(), detail.len());
    let half = approx.len();
    let n = half * 2;
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

    let mut signal = vec![0.0; n];
    for k in 0..half {
        signal[2 * k] = (approx[k] + detail[k]) * inv_sqrt2;
        signal[2 * k + 1] = (approx[k] - detail[k]) * inv_sqrt2;
    }
    signal
}

/// Perform a full multi-resolution Haar wavelet decomposition.
///
/// Decomposes the signal repeatedly until the approximation coefficients
/// reach the minimum size (1). At each level, the approximation is further
/// decomposed while the details are kept.
///
/// # Arguments
/// * `signal` - Input signal as slice of f64 (typically from ternary {-1, 0, +1} values)
///
/// # Panics
/// Panics if signal length is not a power of 2.
pub fn decompose(signal: &[f64]) -> HaarDecomposition {
    let n = signal.len();
    assert!(n > 0, "signal must not be empty");
    assert!(n.is_power_of_two(), "signal length must be a power of 2");

    let mut levels = Vec::new();
    let mut current = signal.to_vec();

    while current.len() >= 2 {
        let (approx, detail) = haar_single_level(&current);
        levels.push(HaarLevel {
            approximation: approx.clone(),
            detail,
        });
        current = approx;
    }

    HaarDecomposition {
        levels,
        original_len: n,
    }
}

/// Reconstruct the original signal from a Haar decomposition.
///
/// Starting from the coarsest approximation, each level is reconstructed by
/// combining the approximation and detail coefficients.
pub fn reconstruct(decomp: &HaarDecomposition) -> Vec<f64> {
    if decomp.levels.is_empty() {
        return Vec::new();
    }

    // Start with the coarsest approximation (last level's approximation)
    let mut current = decomp.levels.last().unwrap().approximation.clone();

    // Reconstruct from coarsest to finest
    for level in decomp.levels.iter().rev() {
        current = inverse_haar_single_level(&current, &level.detail);
    }

    current
}

/// Get all wavelet coefficients as a flat vector in standard order.
///
/// The output is: [detail_level_1, detail_level_2, ..., detail_level_N, final_approximation]
/// This is the standard wavelet coefficient ordering used in compression.
pub fn wavelet_coefficients(decomp: &HaarDecomposition) -> Vec<f64> {
    let mut coeffs = Vec::new();
    for level in &decomp.levels {
        coeffs.extend_from_slice(&level.detail);
    }
    if let Some(last) = decomp.levels.last() {
        coeffs.extend_from_slice(&last.approximation);
    }
    coeffs
}

/// Count the number of coefficients that are exactly (or nearly) zero.
///
/// This measures the sparsity of the wavelet representation, which is important
/// for compression. Ternary signals with constant or slowly-varying regions
/// tend to produce sparse wavelet representations.
pub fn count_zero_coefficients(decomp: &HaarDecomposition, threshold: f64) -> usize {
    let coeffs = wavelet_coefficients(decomp);
    coeffs.iter().filter(|&&c| c.abs() <= threshold).count()
}

/// Apply threshold compression to wavelet coefficients.
///
/// Sets all detail coefficients with magnitude below `threshold` to zero.
/// This is a lossy compression technique: small details are discarded,
/// while significant features are preserved.
///
/// Returns the thresholded decomposition (original is not modified).
pub fn threshold_compress(decomp: &HaarDecomposition, threshold: f64) -> HaarDecomposition {
    let mut new_levels = Vec::new();
    for (i, level) in decomp.levels.iter().enumerate() {
        let is_last = i == decomp.levels.len() - 1;

        let new_approx = if is_last {
            // Keep the final approximation intact
            level.approximation.clone()
        } else {
            // Don't threshold intermediate approximations either (they get decomposed)
            level.approximation.clone()
        };

        let new_detail: Vec<f64> = level
            .detail
            .iter()
            .map(|&d| if d.abs() <= threshold { 0.0 } else { d })
            .collect();

        new_levels.push(HaarLevel {
            approximation: new_approx,
            detail: new_detail,
        });
    }

    HaarDecomposition {
        levels: new_levels,
        original_len: decomp.original_len,
    }
}

/// Compute the compression ratio after thresholding.
///
/// Returns (non_zero_count, total_count, ratio).
/// Ratio = non_zero / total (lower is better compression).
pub fn compression_ratio(decomp: &HaarDecomposition, threshold: f64) -> (usize, usize, f64) {
    let compressed = threshold_compress(decomp, threshold);
    let coeffs = wavelet_coefficients(&compressed);
    let total = coeffs.len();
    let non_zero = coeffs.iter().filter(|&&c| c.abs() > threshold).count();
    let ratio = non_zero as f64 / total as f64;
    (non_zero, total, ratio)
}

/// Compute the reconstruction error (mean squared error) between original and reconstructed signals.
pub fn reconstruction_error(original: &[f64], reconstructed: &[f64]) -> f64 {
    assert_eq!(original.len(), reconstructed.len());
    let n = original.len() as f64;
    original
        .iter()
        .zip(reconstructed.iter())
        .map(|(o, r)| (o - r) * (o - r))
        .sum::<f64>()
        / n
}

/// Compute the energy of a signal (sum of squared values).
pub fn signal_energy(signal: &[f64]) -> f64 {
    signal.iter().map(|v| v * v).sum()
}

/// Compute the energy preserved after compression.
///
/// Returns the ratio of energy in the compressed reconstruction to the original signal energy.
pub fn energy_preserved(original: &[f64], decomp: &HaarDecomposition, threshold: f64) -> f64 {
    let compressed = threshold_compress(decomp, threshold);
    let reconstructed = reconstruct(&compressed);
    if original.is_empty() {
        return 1.0;
    }
    let orig_energy = signal_energy(original);
    if orig_energy < 1e-15 {
        return 1.0;
    }
    signal_energy(&reconstructed) / orig_energy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_level_decomposition() {
        let signal = vec![1.0, -1.0, 0.0, 1.0];
        let (approx, detail) = haar_single_level(&signal);
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        assert_eq!(approx.len(), 2);
        assert_eq!(detail.len(), 2);

        // a[0] = (1 + (-1)) / sqrt(2) = 0
        assert!((approx[0] - 0.0).abs() < 1e-10);
        // a[1] = (0 + 1) / sqrt(2)
        assert!((approx[1] - inv_sqrt2).abs() < 1e-10);
        // d[0] = (1 - (-1)) / sqrt(2) = 2/sqrt(2) = sqrt(2)
        assert!((detail[0] - std::f64::consts::SQRT_2).abs() < 1e-10);
        // d[1] = (0 - 1) / sqrt(2) = -1/sqrt(2)
        assert!((detail[1] + inv_sqrt2).abs() < 1e-10);
    }

    #[test]
    fn test_single_level_roundtrip() {
        let signal = vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        let (approx, detail) = haar_single_level(&signal);
        let recovered = inverse_haar_single_level(&approx, &detail);

        for (i, (orig, rec)) in signal.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-10,
                "Mismatch at {}: {} vs {}",
                i,
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_full_decomposition_size() {
        for &size in &[2, 4, 8, 16, 32, 64] {
            let signal = vec![1.0; size];
            let decomp = decompose(&signal);
            assert_eq!(decomp.original_len, size);
            assert!(!decomp.levels.is_empty());

            let coeffs = wavelet_coefficients(&decomp);
            assert_eq!(coeffs.len(), size, "Total coefficients should equal signal length");
        }
    }

    #[test]
    fn test_reconstruction_matches_original() {
        let signals = vec![
            vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0],
            vec![0.0; 8],
            vec![1.0; 8],
            vec![-1.0; 8],
            vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
        ];

        for signal in &signals {
            let decomp = decompose(signal);
            let recovered = reconstruct(&decomp);

            assert_eq!(recovered.len(), signal.len());
            for (i, (orig, rec)) in signal.iter().zip(recovered.iter()).enumerate() {
                assert!(
                    (orig - rec).abs() < 1e-10,
                    "Reconstruction mismatch at index {}: {} vs {}",
                    i,
                    orig,
                    rec
                );
            }
        }
    }

    #[test]
    fn test_decomposition_levels() {
        // Signal of length 8 -> levels at size 4, 2, 1
        let signal = vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        let decomp = decompose(&signal);

        assert_eq!(decomp.levels.len(), 3);
        assert_eq!(decomp.levels[0].approximation.len(), 4);
        assert_eq!(decomp.levels[0].detail.len(), 4);
        assert_eq!(decomp.levels[1].approximation.len(), 2);
        assert_eq!(decomp.levels[1].detail.len(), 2);
        assert_eq!(decomp.levels[2].approximation.len(), 1);
        assert_eq!(decomp.levels[2].detail.len(), 1);
    }

    #[test]
    fn test_constant_signal_sparse() {
        // A constant signal should have all zero detail coefficients
        let signal = vec![1.0; 8];
        let decomp = decompose(&signal);

        for level in &decomp.levels {
            for (i, &d) in level.detail.iter().enumerate() {
                assert!(
                    d.abs() < 1e-10,
                    "Constant signal should have zero detail at level idx={}, got {}",
                    i,
                    d
                );
            }
        }
    }

    #[test]
    fn test_coefficient_sparsity() {
        // Piecewise constant ternary signal should be sparse
        let signal = vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
        let decomp = decompose(&signal);

        // This signal has a single discontinuity, so most details should be 0
        let zeros = count_zero_coefficients(&decomp, 1e-10);
        let total = wavelet_coefficients(&decomp).len();
        assert!(
            zeros > total / 2,
            "Expected sparse representation: {}/{} zeros",
            zeros,
            total
        );
    }

    #[test]
    fn test_threshold_compression() {
        let signal = vec![1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 1.0, 1.0];
        let decomp = decompose(&signal);

        // With zero threshold, should be lossless
        let compressed = threshold_compress(&decomp, 0.0);
        let recovered = reconstruct(&compressed);
        let error = reconstruction_error(&signal, &recovered);
        assert!(error < 1e-10, "Zero threshold should be lossless, error={}", error);
    }

    #[test]
    fn test_threshold_removes_small_coefficients() {
        let signal = vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
        let decomp = decompose(&signal);

        // With a high threshold, detail coefficients should be zeroed
        let compressed = threshold_compress(&decomp, 1.0);
        for level in &compressed.levels {
            for &d in &level.detail {
                // Either the coefficient was below threshold (now 0) or above (preserved)
                if d.abs() > 0.0 {
                    assert!(d.abs() > 1.0 - 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_compression_ratio() {
        let signal = vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
        let decomp = decompose(&signal);

        let (non_zero, total, ratio) = compression_ratio(&decomp, 0.0);
        assert_eq!(total, 8);
        assert!(ratio <= 1.0);
        assert!(ratio >= 0.0);
        // With zero threshold on a signal with exact zeros in details, some may be exactly 0
        assert!(non_zero <= total);
    }

    #[test]
    fn test_reconstruction_error() {
        let signal = vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        let decomp = decompose(&signal);

        // Lossless
        let recovered = reconstruct(&decomp);
        let error = reconstruction_error(&signal, &recovered);
        assert!(error < 1e-10, "Perfect reconstruction error: {}", error);

        // With compression
        let compressed = threshold_compress(&decomp, 0.5);
        let recovered_compressed = reconstruct(&compressed);
        let error_compressed = reconstruction_error(&signal, &recovered_compressed);
        assert!(error_compressed >= 0.0);
    }

    #[test]
    fn test_signal_energy() {
        let signal = vec![1.0, -1.0, 0.0, 1.0];
        let energy = signal_energy(&signal);
        assert!((energy - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_energy_preserved_lossless() {
        let signal = vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        let decomp = decompose(&signal);
        let preserved = energy_preserved(&signal, &decomp, 0.0);
        assert!(
            (preserved - 1.0).abs() < 1e-9,
            "Lossless should preserve 100% energy, got {}",
            preserved
        );
    }

    #[test]
    fn test_energy_decreases_with_compression() {
        let signal = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
        let decomp = decompose(&signal);

        let preserved_low = energy_preserved(&signal, &decomp, 0.0);
        let preserved_high = energy_preserved(&signal, &decomp, 100.0);

        // Higher threshold should preserve less (or equal) energy
        assert!(
            preserved_high <= preserved_low + 1e-9,
            "Higher threshold should preserve less energy: {} vs {}",
            preserved_high,
            preserved_low
        );
    }

    #[test]
    fn test_zero_signal() {
        let signal = vec![0.0; 8];
        let decomp = decompose(&signal);

        for level in &decomp.levels {
            for &a in &level.approximation {
                assert!(a.abs() < 1e-10, "Zero signal should have zero approx");
            }
            for &d in &level.detail {
                assert!(d.abs() < 1e-10, "Zero signal should have zero detail");
            }
        }
    }

    #[test]
    fn test_alternating_signal() {
        let signal = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let decomp = decompose(&signal);

        // Alternating signal has maximum detail at the finest level
        let first_level_detail_energy: f64 =
            decomp.levels[0].detail.iter().map(|d| d * d).sum();
        assert!(
            first_level_detail_energy > 0.0,
            "Alternating signal should have detail energy"
        );
    }

    #[test]
    fn test_multi_resolution_energy_conservation() {
        // Energy at each level should be conserved (Parseval-like for Haar)
        let signal = vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        let decomp = decompose(&signal);

        let original_energy = signal_energy(&signal);

        // Total wavelet coefficient energy should equal original signal energy
        let coeffs = wavelet_coefficients(&decomp);
        let coeff_energy = signal_energy(&coeffs);

        assert!(
            (original_energy - coeff_energy).abs() < 1e-9,
            "Energy conservation: original={}, coefficients={}",
            original_energy,
            coeff_energy
        );
    }

    #[test]
    fn test_large_signal() {
        let mut signal = Vec::with_capacity(256);
        for i in 0..256 {
            signal.push(match i % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            });
        }

        let decomp = decompose(&signal);
        let recovered = reconstruct(&decomp);

        for (i, (orig, rec)) in signal.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-8,
                "Large signal mismatch at {}: {} vs {}",
                i,
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_step_function_compression() {
        // Step function: perfect for Haar (should compress very well)
        let mut signal = Vec::new();
        for _ in 0..64 {
            signal.push(1.0);
        }
        for _ in 0..64 {
            signal.push(-1.0);
        }
        for _ in 0..64 {
            signal.push(0.0);
        }
        for _ in 0..64 {
            signal.push(1.0);
        }

        let decomp = decompose(&signal);

        // With moderate threshold, should still reconstruct well
        let compressed = threshold_compress(&decomp, 0.01);
        let recovered = reconstruct(&compressed);
        let error = reconstruction_error(&signal, &recovered);

        assert!(
            error < 0.01,
            "Step function should compress well, error={}",
            error
        );
    }

    #[test]
    #[should_panic(expected = "signal length must be a power of 2")]
    fn test_decompose_non_power_of_2() {
        decompose(&[1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "signal must not be empty")]
    fn test_decompose_empty() {
        decompose(&[]);
    }
}
