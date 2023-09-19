use faer_core::{mul::matmul, Mat, MatRef};

use crate::utils::*;

/*
last two functions to be made
*/
//fitness function parameters are the k-forms, and the k+1 forms
pub fn rank_pop(
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
    for j in 1..k_forms.len() {
        println!("calculating for k-form {}", j);
        ranked_pop.push(compute_fitness(
            k_forms[j].as_ref(),
            k_forms[j - 1].as_ref(),
            k_plus_one_forms[j].as_ref(),
            ext_d_population[j].as_ref(),
        ));
        println!("-------------");
    }
    println!("Done");
    return ranked_pop;
}

//fitness will take in a k-form, another k-form, a k+1-form and an exterior derivative
pub fn compute_fitness(
    k_form_a: MatRef<f64>,
    k_form_b: MatRef<f64>,
    k_plus_one_form: MatRef<f64>,
    exterior_derivative: MatRef<f64>,
) -> (Mat<f64>, f64) {
    let mut fitness: f64 = 0.0f64;
    let mut closeness: f64 = 0.0f64;
    //numerical closeness between dw and w_p+1:
    let mut mulbuff = Mat::zeros(k_form_a.nrows(), k_form_a.ncols());
    matmul(
        mulbuff.as_mut(),
        exterior_derivative,
        k_form_a,
        None,
        1.0f64,
        faer_core::Parallelism::Rayon((0)),
    );
    for y in 0..mulbuff.nrows() {
        for x in 0..mulbuff.ncols() {
            println!(
                "k_plus_one form {} {} {}",
                y,
                x,
                k_plus_one_form.read(y, x).powf(2.0f64)
            );
            println!("mulbuff {} {} {}", y, x, mulbuff.read(y, x).powf(2.0f64));

            closeness += exp(- (k_plus_one_form.read(y, x).powf(2.0f64)
                - mulbuff.read(y, x).powf(2.0f64)))
            .abs()
            .sqrt(); //will determine the numerical closeness of each element of each matrix
        }
    }

    //computes the nilpotency of the map
    let mut nilp_metric = 0;
    let mut nilp_buff: Mat<f64> =
        Mat::zeros(exterior_derivative.nrows(), exterior_derivative.ncols());
    matmul(
        nilp_buff.as_mut(),
        exterior_derivative,
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
    let a = k_form_a.to_owned();
    let b: Mat<f64> = k_form_b.to_owned();
    let wedge_buff = wedge_m(&a, &b);
    let mut ext_div_buff_l = Mat::zeros(wedge_buff.nrows(), wedge_buff.ncols());
    let ext_d_pop_buff = pad_for_proc(&exterior_derivative.to_owned(), &ext_div_buff_l);
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
    let buff_d = pad_for_proc(&exterior_derivative.to_owned(), &da);
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
    let a_rref = gaussian_elimination(&a);
    let p = compute_dims(&a_rref);
    let antisymmetry_coefficient: i32 = -1;
    let a_wedge_db_buff = wedge_m(&a, &db);
    let a_wedge_db = Mat::with_dims(a_wedge_db_buff.nrows(), a_wedge_db_buff.ncols(), |i, j| {
        antisymmetry_coefficient.pow(p) as f64 * a_wedge_db_buff.read(i, j)
    });

    let mut ext_div_buff_r = wedge_m(&da, &b) + a_wedge_db;

    println!(
        "LHS {}, RHS {}",
        ext_div_buff_r.ncols(),
        ext_div_buff_l.ncols()
    );
    //force dimensional equality:
    if (ext_div_buff_r.ncols() < ext_div_buff_l.ncols()) {
        ext_div_buff_r = pad_for_proc(&ext_div_buff_r, &ext_div_buff_l);
    } else if (ext_div_buff_r.ncols() > ext_div_buff_l.ncols()) {
        ext_div_buff_l = pad_for_proc(&ext_div_buff_l, &ext_div_buff_r);
    }
    //now determine fitness of this equality:
    let ext_div_result = ext_div_buff_r - ext_div_buff_l;
    let mut derivative_closeness: i64 = 0;

    for y in 0..ext_div_result.nrows() as usize {
        for x in 0..ext_div_result.ncols() as usize {
            derivative_closeness = derivative_closeness
                .saturating_add(ext_div_result.read(y, x).powf(2.0f64).sqrt() as i64);
        }
    }
    println!(
        "d closeness {}, nilp metric {}, closeness {}",
        derivative_closeness, nilp_metric, closeness
    );
    fitness = derivative_closeness as f64 + nilp_metric as f64 + (closeness as f64).powf(-1.0f64);

    let tuple = (exterior_derivative.to_owned(), fitness);
    return tuple;
}
