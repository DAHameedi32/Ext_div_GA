use faer_core::{mul::matmul, Mat};
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

/*
last two functions to be made
*/
//fitness function parameters are the k-forms, and the k+1 forms
pub fn fitness(
    k_forms: &Vec<Mat<f64>>,
    k_plus_one_forms: &Vec<Mat<f64>>,
    ext_d_population: &Vec<Mat<f64>>,
) -> Vec<(Mat<f64>, f64)> {
    let mut ranked_pop: Vec<(Mat<f64>, f64)> = vec![];
    //Axioms:
    //Exterior Derivative property: d(a ∧ b) = (da ∧ b) + (-1)^p (a ∧ db) where a and b are p-forms
    //Nilpotency: ddw = 0
    //is it in the kernel dw = 0? if so then the above is trivial
    //closeness of dw_p to w_p+1
    //
    //derivative property i guess?
    //questionable how to implement this
    for j in 0..k_forms.len() {
        for i in 0..ext_d_population.len() {
            let ext_d_buff = ext_d_population[i].as_ref();
            let mut fitness: f64 = 0.0f64;
            let mut closeness: f64 = 0.0f64;
            //numerical closeness between dw and w_p+1:
            let mut mulbuff = Mat::zeros(k_forms[j].nrows(), k_forms[j].ncols());
            matmul(
                mulbuff.as_mut(),
                ext_d_population[i].as_ref(),
                k_forms[j].as_ref(),
                None,
                1.0f64,
                faer_core::Parallelism::Rayon((0)),
            );
            for y in 0..mulbuff.nrows() {
                for x in 0..mulbuff.ncols() {
                    closeness += (k_plus_one_forms[j].read(y, x).powf(2.0f64)
                        - mulbuff.read((y), (x)).powf(2.0f64))
                    .sqrt(); //will determine the numerical closeness of each element of each matrix
                }
            }

            //computes the nilpotency of the map
            let mut nilp_metric = 0;
            let mut nilp_buff: Mat<f64> =
                Mat::zeros(ext_d_population[i].nrows(), ext_d_population[i].ncols());
            matmul(
                nilp_buff.as_mut(),
                ext_d_population[i].as_ref(),
                mulbuff.as_ref(),
                None,
                1.0f64,
                faer_core::Parallelism::Rayon(0),
            );
            for y in 0..nilp_buff.nrows() {
                for x in 0..nilp_buff.ncols() {
                    if (nilp_buff.read(y, x) == 0.0) {
                        //adds large number to fitness
                        nilp_metric += 10;
                    } else if (nilp_buff.read(y, x) != 0.0) {
                        //subtracts from fitness
                        nilp_metric -= 10;
                    }
                }
            }

            //Ext_div property:
            //d(a ∧ b) = (da ∧ b) + (-1)^p (a ∧ db) where a and b are p-forms
            //takes the k-forms a and b, and wedges them together: need to ensure that the dimensionality is okay:
            //do this by getting the jth k-form and the j+1th k-form and padding them out
            let a = k_forms[j].clone();
            let b: Mat<f64>;
            if j + 1 > k_forms.len() {
                b = k_forms[0].clone();
            } else {
                b = k_forms[j + 1].clone();
            }
            let wedge_buff = wedge_m(&a, &b);
            let mut ext_div_buff_l = Mat::zeros(wedge_buff.nrows(), wedge_buff.ncols());
            let ext_d_pop_buff = pad_for_proc(&ext_d_population[i], &ext_div_buff_l);
            matmul(
                ext_div_buff_l.as_mut(),
                ext_d_pop_buff.as_ref(),
                wedge_buff.as_ref(),
                None,
                1.0f64,
                faer_core::Parallelism::Rayon((0)),
            );

            let mut da: Mat<f64> = Mat::zeros(ext_div_buff_l.nrows(), ext_div_buff_l.ncols());
            let mut db: Mat<f64> = Mat::zeros(ext_div_buff_l.nrows(), ext_div_buff_l.ncols());
            let buff_d = pad_for_proc(&ext_d_population[i], &da);
            let buff_b = pad_for_proc(&b, &db);
            let buff_a = pad_for_proc(&a, &da);
            matmul(
                da.as_mut(),
                buff_d.as_ref(),
                buff_a.as_ref(),
                None,
                1.0f64,
                faer_core::Parallelism::Rayon(0),
            );
            matmul(
                db.as_mut(),
                buff_d.as_ref(),
                buff_b.as_ref(),
                None,
                1.0f64,
                faer_core::Parallelism::Rayon(0),
            );
            let a_rref = gaussian_elimination(a.to_owned());
            let p = compute_dims(&a_rref);
            let antisymmetry_coefficient: i32 = -1;
            let a_wedge_db_buff = wedge_m(&a, &db);
            let a_wedge_db =
                Mat::with_dims(a_wedge_db_buff.nrows(), a_wedge_db_buff.ncols(), |i, j| {
                    antisymmetry_coefficient.pow(p) as f64 * a_wedge_db_buff.read(i, j)
                });

            let ext_div_buff_r = wedge_m(&da, &b) + a_wedge_db;

            println!(
                "LHS {}, RHS {}",
                ext_div_buff_r.ncols(),
                ext_div_buff_l.ncols()
            );
            //now determine fitness of this equality:
            let ext_div_result = ext_div_buff_r - ext_div_buff_l; // < -- error
            let mut derivative_closeness = 0;

            for y in 0..ext_div_result.nrows() as usize {
                for x in 0..ext_div_result.ncols() as usize {
                    derivative_closeness += ext_div_result.read(y, x).powf(2.0f64).sqrt() as i64;
                }
            }
            fitness = derivative_closeness as f64 + nilp_metric as f64 + closeness as f64;

            let tuple = (ext_d_buff.to_owned(), fitness);
            ranked_pop.push(tuple);
        }
    }
    return ranked_pop;
}

pub fn mat_pad(a: Mat<f64>, b: Mat<f64>)
//takes in 2 matrices and returns both matrices padded to the larger one
{
    if (a.ncols() != b.ncols()) {
        if (a.ncols() < b.ncols()) {
            //expands a to be same column size as b
        }
        if (a.ncols() > b.ncols()) {
            //expands b to be same column size as a
        }
    }
    if (a.nrows() != b.nrows()) {
        if (a.nrows() < b.nrows()) {
            //expands a to same rowsize as b
        }
        if (a.nrows() > b.nrows()) {
            //expands b to same rowsize as a
        }
    }
}

pub fn GA_main(
    k_forms: Vec<Mat<f64>>,
    k_plus_one_forms: Vec<Mat<f64>>,
    _generations: i8,
    _pop_size: usize,
) -> (Mat<f64>, f64) {
    let mut population = pop_init(_pop_size, &k_plus_one_forms[0]); //initialises population of exterior derivative matrices

    //repeat this block of code ad 100 generations
    for _r in 0.._generations {
        let pop_buffer = population.clone();
        let moved_k_forms = k_forms.clone();
        let moved_k_plus_one_forms = k_plus_one_forms.clone();

        let ranked_pop = fitness(&moved_k_forms, &moved_k_plus_one_forms, &pop_buffer); // current error here
                                                                                        // panic!("fitness done");
        let mean_fit;
        let mut total_fit = 0.0f64;
        let mut counter = 0;
        for k in 0..ranked_pop.len() {
            total_fit += ranked_pop[k].1;
            counter += 1;
        }
        mean_fit = total_fit / counter as f64;

        let mut breeding_pairs: Vec<(Mat<f64>, Mat<f64>)> = vec![];
        let mut new_gen: Vec<Mat<f64>> = vec![];
        //find pairs for breeding
        for i in 0..ranked_pop.len() {
            for j in 0..ranked_pop.len() {
                if j != i {
                    //need to make a metric to calculate whether 2 matrices can breed
                    let _reproduction_metric = ranked_pop[i].1 as f64 + ranked_pop[j].1 as f64;
                    if (_reproduction_metric >= 1.5 * mean_fit) {
                        let _pair_tuple = (ranked_pop[i].0.clone(), ranked_pop[j].0.clone());
                        breeding_pairs.push(_pair_tuple);
                    }
                }
            }
        }

        //now iterate through the breeding pairs and create a new generation:
        for i in 0..breeding_pairs.len() {
            new_gen.push(reproduce(&breeding_pairs[i].0, &breeding_pairs[i].1));
        }
        //check to see how large the new generation is:
        //if larger than pop_size then cull at random
        //if lower than pop_size then bring in members of older population:
        if (new_gen.len() > _pop_size as usize) {
            while ((new_gen.len() as i64) > _pop_size as i64) {
                //cull at random:
                let mut rng = rand::thread_rng();
                let x: i64 = rng.gen_range(0..new_gen.len() as i64);
                new_gen.remove(x as usize);
            }
        }
        if (new_gen.len() < _pop_size as usize) {
            //find how many more members are needed:
            let perc_fill: f64 = new_gen.len() as f64 / _pop_size as f64;
            let prob_fill = 1.0f64 - perc_fill;
            for i in 0.._pop_size as usize {
                //populate with matrices from old gen:
                let mut rng = rand::thread_rng();
                let x: f64 = rng.gen();
                if (x > prob_fill) {
                    new_gen.push(pop_buffer[i].to_owned());
                }
            }
        }
        //once all these checks have been fulfilled, then we can mutate the new generation:
        for i in 0..new_gen.len() {
            let prob_mut = 0.05f64;
            let mut rng = rand::thread_rng();
            let x: f64 = rng.gen();
            if (x > prob_mut) {
                let matrix_buff: Mat<f64> = Mat::zeros(new_gen[i].nrows(), new_gen[i].ncols());
                let bit_string = to_bitstring(&new_gen[i]);
                let new_bit_string = mutate(bit_string, prob_mut);
                let new_mat = mat_from_bitstring(new_bit_string, matrix_buff);
                new_gen[i] = new_mat;
            }
        }
        //let population = new_gen:
        population = new_gen;
        //repeat for 100 generations now
    }
    //finally extract the fittest matrix and return it:
    let final_pop = fitness(&k_forms, &k_plus_one_forms, &population);
    let mut fittest_index = 0;
    let mut fitness_value = 0.0f64;
    for i in 0..final_pop.len() as usize {
        if (final_pop[i].1 >= fitness_value) {
            fitness_value = final_pop[i].1;
            fittest_index = i;
        }
    }
    return final_pop[fittest_index].clone();
}

//define a wedge product:
//vectors as a matrix: u, v.
//Wedge product is defined as: (u^Tv - uv^T) for row vectors Ref: https://www.math.purdue.edu/~arapura/preprints/diffforms.pdf
//taken as given that u and v have same dimension
//however this only works for vectors, so could only generate 2-forms and maybe 3-forms

//need to define a function to take the rank 3 tensor to a rank 2 tensor
pub fn flatten_to_mat(tensor: Vec<Vec<Mat<f64>>>) -> Mat<f64> {
    let y_dim_outer = tensor.len();
    println!("{}", y_dim_outer);
    let x_dim_outer = tensor[0].len();
    println!("{}", x_dim_outer);
    let y_dim_inner = tensor[0][0].ncols();
    println!("{}", y_dim_inner);
    let x_dim_inner = tensor[0][0].nrows();
    println!("{}", x_dim_inner);

    let fin_dims_y = (y_dim_outer * y_dim_inner); // (y_dim_outer * y_dim_inner);
    let fin_dims_x = (x_dim_outer * x_dim_inner);
    println!("{}", fin_dims_y);
    println!("{}", fin_dims_x);
    let mut fin_mat: Mat<f64> = Mat::zeros((fin_dims_y) as usize, (fin_dims_x) as usize);
    for y_index in 0..y_dim_outer as usize {
        for x_index in 0..x_dim_outer as usize {
            for i_index in 0..y_dim_inner {
                for j_index in 0..x_dim_inner {
                    fin_mat.write(
                        y_index * y_dim_outer + i_index,
                        x_index * x_dim_outer + j_index,
                        tensor[y_index][x_index].read(i_index, j_index),
                    );
                }
            }
        }
    }
    println!("---------------------------------------------------------------------------------------------------");
    return fin_mat;
}

/// Given A, an  m x n matrix and B a p x q matrix
/// Return the Kronecker product, a pm x qn matrix
pub fn direct_prod(A: &Mat<f64>, B: &Mat<f64>) -> Mat<f64> {
    // println!("B {:?}", B.read(0, 0));
    // println!("Direct");
    //fetch data about A
    let m = A.nrows();
    let n = A.ncols();

    //fetch data about B
    let p = B.nrows();
    let q = B.ncols();

    // panic!("here");
    let ret_mat = Mat::with_dims(p * m, q * n, |i, j| {
        A.read(i / p, j / q) * B.read(i % p, j % q)
    });

    return ret_mat; //returns a matrix
}

pub fn wedge_v(u: &Mat<f64>, v: &Mat<f64>) -> Mat<f64> //wedge product for vectors
{
    //This is the process for vectors

    //remember to as_ref() row vectors when you call the function
    //need to fill out matrices for the associated vectors, as they are both n-dimensional row vectors we convert to an nxn matrix with the vector taking up the first row, 0 in all other entries
    let mut u_buf = Mat::zeros(u.ncols(), u.ncols());
    let mut v_buf = Mat::zeros(v.ncols(), v.ncols());

    for i in 0..u_buf.ncols() {
        u_buf.write(0, i, u.read(0, i));
    }
    for j in 0..v_buf.ncols() {
        v_buf.write(0, j, v.read(0, j));
    }
    //create some buffer matrices to store the product:
    let mut buff_l = Mat::zeros(u_buf.nrows(), v_buf.ncols());
    let mut buff_r = Mat::zeros(u_buf.nrows(), v_buf.ncols());

    matmul(
        buff_l.as_mut(),
        u_buf.transpose(),
        v_buf.as_ref(),
        None,
        1.0f64,
        faer_core::Parallelism::Rayon(0),
    ); //computes u^Tv sends -> buff_l
    matmul(
        buff_r.as_mut(),
        u_buf.as_ref(),
        v_buf.transpose(),
        None,
        1.0f64,
        faer_core::Parallelism::Rayon(0),
    ); //computes uv^T sends -> buff_r

    //compute subtraction in brackets:
    let brack_ex = buff_l - buff_r; //stores final bracketed expression: (u^Tv - uv^T)
    let identity = Mat::with_dims(brack_ex.nrows(), brack_ex.ncols(), |i, j| {
        if i == j {
            1.0
        } else {
            0.0
        }
    }); //generate identity matrix for use in wedge computation

    let mut wedge_prod_fin = Mat::zeros(u.ncols(), v.ncols()); //store final wedge product
    matmul(
        wedge_prod_fin.as_mut(),
        identity.as_ref(),
        brack_ex.as_ref(),
        None,
        1.0f64,
        faer_core::Parallelism::Rayon(0),
    );

    return wedge_prod_fin; //returns wedge_product matrix
}

pub fn wedge_m(u: &Mat<f64>, v: &Mat<f64>) -> Mat<f64> //wedge product for matrices
{
    let fin_mat = direct_prod(u, v) - direct_prod(v, u);
    return fin_mat;
}

pub fn compute_dims(u: &Mat<f64>) -> u32 {
    let mut dimensionality: u32 = 0;
    for y in 0..u.nrows() {
        for x in 0..u.ncols() {
            if (u.read(y, x) != 0.0f64) {
                dimensionality += 1;
            }
        }
    }
    return dimensionality;
}

pub fn pad_for_proc(targ_mat: &Mat<f64>, sample_mat: &Mat<f64>) -> Mat<f64> {
    let sample_x = sample_mat.ncols();
    let sample_y = sample_mat.nrows();

    let mut return_mat: Mat<f64> = Mat::zeros(sample_y, sample_x);
    for i in 0..targ_mat.nrows() {
        for j in 0..targ_mat.ncols() {
            return_mat.write(i, j, targ_mat.read(i, j));
        }
    }
    return return_mat;
}

pub fn gaussian_elimination(mut matrix: Mat<f64>) -> Mat<f64> {
    let rows = matrix.nrows();
    if rows == 0 {
        return matrix; // Nothing to do for an empty matrix.
    }
    let cols = matrix.ncols();

    let mut lead = 0; // The column we are working on.

    for r in 0..rows {
        if lead >= cols {
            break;
        }

        let mut i = r;
        while matrix.read(i, lead) == 0.0 {
            i += 1;
            if i == rows {
                i = r;
                lead += 1;
                if cols == lead {
                    return matrix; // We are done, no more pivots to find.
                }
            }
        }

        // Swap rows i and r.
        //take rows i and r then assign them to separate vectors:
        let mut row_i_buf: Vec<f64> = vec![];
        let mut row_r_buf: Vec<f64> = vec![];
        //populate the buffers:
        for i_col_iterator in 0..matrix.ncols() {
            row_i_buf.push(matrix.read(i, i_col_iterator));
        }
        for r_col_iterator in 0..matrix.ncols() {
            row_r_buf.push(matrix.read(r, r_col_iterator));
        }
        for i_row_iterator in 0..matrix.nrows() {
            matrix.write(i, i_row_iterator, row_i_buf[i_row_iterator]);
        }
        for r_row_iterator in 0..matrix.nrows() {
            matrix.write(r, r_row_iterator, row_r_buf[r_row_iterator]);
        }
        // Scale the pivot row (r) to have a 1 at the pivot position.
        let pivot = matrix.read(r, lead);
        for j in 0..cols {
            let div = matrix.read(r, j) / pivot;
            matrix.write(r, j, div);
        }

        // Eliminate other rows.
        for i in 0..rows {
            if i != r {
                let factor = matrix.read(i, lead);
                for j in 0..cols {
                    let sub = matrix.read(i, j) - (factor * matrix.read(r, j));
                    matrix.write(i, j, sub);
                }
            }
        }

        lead += 1;
    }

    return matrix;
}
