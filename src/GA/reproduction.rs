use faer_core::Mat;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Generate a random population of k+1 forms
pub fn pop_init(pop_size: usize, k_plus_one_form: &Mat<f64>) -> Vec<Mat<f64>> {
    let mut pop_vec: Vec<Mat<f64>> = Vec::with_capacity(pop_size);

    // seed the PRNG
    let mut rng = StdRng::from_seed([0; 32]);

    //populate the population vector
    for i in 0..pop_size {
        let individual: Mat<f64> =
            Mat::with_dims(k_plus_one_form.nrows(), k_plus_one_form.ncols(), |_, _| {
                rng.gen()
            });
        pop_vec.push(individual);
    }
    return pop_vec;
}

pub fn reproduce(father: &Mat<f64>, mother: &Mat<f64>) -> Mat<f64> {
    let fin_row = father.nrows();
    let fin_col = father.ncols();
    let father_genome: Vec<String> = to_bitstring(&father);
    let mother_genome: Vec<String> = to_bitstring(&mother);
    println!("{:?}", father_genome);
    println!("{:?}", mother_genome);

    let mut child_genome: Vec<String> = vec![];
    let mut gen_length = 0;
    //iterate through genomes and collect bit string into char vectors
    for y in 0..father_genome.len() {
        let mut c_g_chars: Vec<char> = vec![];
        let f_g_chars: Vec<_> = father_genome[y].chars().collect();
        let m_g_chars: Vec<_> = mother_genome[y].chars().collect(); //collects the matrix element genome into a vector<char>
                                                                    //can now go inside the chars array which has just been made and assign bits to the child genome as required:
        for j in 0..f_g_chars.len()
        //collects a float as individual bits
        {
            //generate random number
            let mut rng = rand::thread_rng();
            let x: f64 = rng.gen();
            if (x >= 0.5f64) {
                //take the bit from father genome
                c_g_chars.push(f_g_chars[j]);
            }
            if (x < 0.5) {
                //take the bit from mother genome
                c_g_chars.push(m_g_chars[j]);
            }
        }
        //now need to format the genome:
        //collect c_g_chars as one string:
        child_genome.push(c_g_chars.iter().collect());
    }
    println!("{:?}", child_genome);

    let mut ch: Mat<f64> = Mat::zeros(fin_row, fin_col);
    let child = mat_from_bitstring(child_genome, ch);
    println!("{:?}", child);
    return child;
}

pub fn to_bitstring(m: &Mat<f64>) -> Vec<String> {
    let mut bit_string: Vec<String> = vec![];
    let m_range_y = m.nrows() as usize;
    let m_range_x = m.ncols() as usize;

    for y in 0..m_range_y {
        for x in 0..m_range_x {
            let element = m.read(y, x);
            let binary_representation = element.to_bits();
            println!("{:#?}", binary_representation);
            let bits = format!("{:065b}", binary_representation).chars().collect();
            bit_string.push(bits);
        }
    }
    return bit_string;
}

pub fn mutate(bit_string: Vec<String>, mutation_probability: f64) -> Vec<String> {
    let mut res_vec = vec![];
    for i in 0..bit_string.len() {
        let mut bit_sep: Vec<char> = bit_string[i].chars().collect();
        for j in 0..bit_sep.len() {
            let mut rng = rand::thread_rng();
            let x: f64 = rng.gen();
            if (x <= mutation_probability) {
                if (bit_sep[j] == '0') {
                    bit_sep[j] = '1';
                } else if (bit_sep[j] == '1') {
                    bit_sep[j] = '0';
                }
            }
        }
        let final_string = bit_sep.into_iter().collect();
        res_vec.push(final_string);
    }
    return res_vec;
}

pub fn mat_from_bitstring(bit_string: Vec<String>, mut targ_mat: Mat<f64>) -> Mat<f64> {
    let targ_mat_range_y = targ_mat.nrows() as usize;
    let targ_mat_range_x = targ_mat.ncols() as usize;

    //create an array to store the float:
    let mut float_array: Vec<f64> = vec![];
    let bs = bit_string;
    //convert the bit strings to f64s:
    for i in 0..bs.len() {
        let binary_rep = &bs[i];
        let float_bits = u64::from_str_radix(&binary_rep, 2).expect("Error!");
        let f64_value = f64::from_bits(float_bits);
        float_array.push(f64_value);
    }
    //now add a nested for loop: to populate the target matrix:
    let mut matrix_iterator = 0;
    for y in 0..targ_mat_range_y {
        for x in 0..targ_mat_range_x {
            targ_mat.write(y, x, float_array[x + matrix_iterator * y]);
        }
        matrix_iterator += 1;
    }
    return targ_mat;
}
