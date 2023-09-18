#![allow(non_snake_case, clippy::needless_return)]

use faer_core::{mat, Mat};

mod GA;
mod utils;
fn main() {
    /*
    //initial functions in space are numerically described as an evolution from initial conditons.
    //we can define the 1-form as being the vector rep of the function in space:
    //F(x,y,z,...)dx + G(x,y,z,...)dy+...
    //is equivalent to F(...)i + G(...)j +...
    // let dims : usize = 3; //dimensions of our space
    //give the explicit 1-forms and the initial conditions we can construct the k-forms as necessary
    //can compute the functions explicitly:
    /*
    let F = 9 * x_1 + 5;
    let G = 6 * x_2 + 3;
    let H = 3 * x_3 + 2;
     */

    //...
    //define some restrictions on the problem:
    //important data for numerical construction:
    /*
    let starting_x_1 = 0;
    let final_x_1 = 5;
    let step_x_1 = 0.1;

    let starting_x_2 = 0;
    let final_x_2 = 5;
    let step_x_2 = 0.1;

    let starting_x_3 = 0;
    let final_x_3 = 5;
    let step_x_3 = 0.1;
    */
    //constructing the functions:
    */
    //once we compute the functions in F(), G(), ... we can compute the wedge at all points
    let p = mat!([2.0f64, 0.0f64, 0.0f64, 0.0f64]); //these row vectors will contain the output of the explicit functions
    let q = mat!([0.0f64, 2.0f64, 0.0f64, 0.0f64]);
    let r = mat!([0.0f64, 0.0f64, 2.0f64, 0.0f64]);
    let s = mat!([0.0f64, 0.0f64, 0.0f64, 2.0f64]);
    let one_forms = vec![p, q, r, s];
    //generate all possible 2-forms from the given 1-forms
    let mut two_forms: Vec<Mat<f64>> = vec![];
    for i in 0..one_forms.len() {
        for j in 0..one_forms.len() {
            two_forms.push(utils::wedge_v(&one_forms[i], &one_forms[j]));
        }
    }

    //generate all possible 3-forms from the given 2-forms and 1-forms
    let mut three_forms: Vec<Mat<f64>> = vec![];
    for i in 0..one_forms.len() {
        for j in 0..one_forms.len() {
            let one_form_buff = utils::pad_for_proc(&one_forms[i], &two_forms[j]);
            three_forms.push(utils::wedge_v(&one_form_buff, &two_forms[j]));
        }
    }

    let ext_div = GA::GA_main(two_forms, three_forms, 100, 100);
    println!("{:?}", ext_div);
    /*
    let c: Mat<f64> = wedge(p.as_ref(), q.as_ref()); //c is a 2-form
    let k: Mat<f64> = wedge(r.as_ref(), c.as_ref()); //k should be a 3-form as wedge product is associative
    let K: Mat<f64> = wedge(s.as_ref(), k.as_ref()); //should be a 4-form
    println!("{:?}", c);
    println!("{:?}", k);
    println!("{:?}", K);
    */
}
