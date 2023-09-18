use faer_core::Mat;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Generate a random population of k+1 forms
pub fn pop_init(pop_size: usize, k_plus_one_form: &Mat<f64>) -> Vec<Mat<f64>> {
    let mut pop_vec: Vec<Mat<f64>> = Vec::with_capacity(pop_size);

    // seed the PRNG
    let mut rng = StdRng::from_seed([0; 32]);

    //populate the population vector
    for _ in 0..pop_size {
        let individual: Mat<f64> =
            Mat::with_dims(k_plus_one_form.nrows(), k_plus_one_form.ncols(), |_, _| {
                rng.gen()
            });
        pop_vec.push(individual);
    }
    return pop_vec;
}

/// Given father bit a, mother bit b and mask m where if m = 1 take father else take mother
/// a | b | m || output
/// 0 | 0 | 0 || 0
/// 0 | 0 | 1 || 0
/// 0 | 1 | 0 || 1
/// 0 | 1 | 1 || 0
/// 1 | 0 | 0 || 0
/// 1 | 0 | 1 || 1
/// 1 | 1 | 0 || 1
/// 1 | 1 | 1 || 1
///
/// equivalent to ((a AND m) OR (b AND NOT m)) , so can be done bitwise
pub fn reproduce(father: &Mat<f64>, mother: &Mat<f64>) -> Mat<f64> {
    if father.ncols() != mother.ncols() || father.nrows() != mother.nrows() {
        panic!("input matrices have different dimensions")
    }
    let mut rng = rand::thread_rng();
    let child: Mat<f64> = Mat::with_dims(father.nrows(), father.ncols(), |row, col| {
        let mut f_bytes = father.read(row, col).to_ne_bytes(); // mutate in place to save memory
        let m_bytes = mother.read(row, col).to_ne_bytes();
        f_bytes
            .iter_mut()
            .zip(m_bytes)
            .for_each(|(f_byte, m_byte)| {
                let mask: u8 = rng.gen();
                *f_byte = (*f_byte & mask) | (m_byte & !mask);
            });
        f64::from_ne_bytes(f_bytes)
    });
    return child;
}

/// Mutate the element of the vector bitwise
/// Given mask where 0 is do nothing and 1 is flip
/// initial | mask || output
///    0    |   0  ||   0
///    0    |   1  ||   1
///    1    |   0  ||   1
///    1    |   1  ||   0
///
/// ie bitwise XOR
pub fn mutate(matrix: &Mat<f64>, mutation_probability: f64) -> Mat<f64> {
    let mut rng = rand::thread_rng();
    let mutated_matrix: Mat<f64> = Mat::with_dims(matrix.nrows(), matrix.ncols(), |row, col| {
        let x = matrix.read(row, col);
        let mut x_bytes = x.to_ne_bytes();
        x_bytes.iter_mut().for_each(|byte| {
            let mut mask = 0u8;
            for i in 0..7 {
                let out: f64 = rng.gen();
                if out <= mutation_probability {
                    mask += 2u8.pow(i);
                }
            }
            *byte = *byte ^ mask // do bitwise XOR
        });
        // turn back into integer
        f64::from_ne_bytes(x_bytes)
    });
    return mutated_matrix;
}

#[cfg(test)]
mod tests {
    use crate::GA::reproduction::*;
    use faer_core::mat;
    #[test]
    fn rep_mutate_standard() {
        let a = mat![[1., 2.], [3., 4.]];
        let b = mat![[0., 5.], [6., 7.]];
        let c = mat![
            [0., 5., 0., 10.],
            [6., 7., 12., 14.],
            [0., 15., 0., 20.],
            [18., 21., 24., 28.],
        ];
        reproduce(&a, &b);
        mutate(&a, 1.);
        mutate(&b, 1.);
        mutate(&c, 1.);
    }
    #[test]
    #[should_panic]
    fn rep_panic() {
        let a = mat![[1., 2.], [3., 4.]];
        let c = mat![
            [0., 5., 0., 10.],
            [6., 7., 12., 14.],
            [0., 15., 0., 20.],
            [18., 21., 24., 28.],
        ];
        reproduce(&a, &c);
    }
}
