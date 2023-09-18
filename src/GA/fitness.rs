use faer_core::{mul::matmul, Mat};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::utils::*;

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
