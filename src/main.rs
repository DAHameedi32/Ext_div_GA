use faer_core::{Mat, mul::matmul, MatMut, MatRef};

// to do: learn enough about de Rham Cohomology to understand it and the exterior derivative map
//inputs a function
//converts to a 1-form

//define a wedge product:
//vectors as a matrix: u, v.
//Wedge product is defined as: 1/2 (u^Tv - uv^T) 
//taken as given that u and v have same dimension
pub fn wedge(u : MatRef<f64>, v : MatRef<f64>) -> Mat<f64>{
    let brack_ex_l : MatMut<f64>; //stores left bracket expression: (u^Tv)
    let brack_ex_r : MatMut<f64>; //stores right bracket expression: (uv^T)
     //stores final bracketed expression: (u^Tv - uv^T)

    matmul(brack_ex_l, u.transpose(), v, None, 1.0f64, faer_core::Parallelism::Rayon(0)); //computes u^Tv sends -> brack_ex_l
    matmul(brack_ex_r, u, v.transpose(), None, 1.0f64, faer_core::Parallelism::Rayon(0)); //computes uv^T sends -> brack_ex_r
    
    //compute subtraction in brackets:

    let brack_ex = brack_ex_l.to_owned() - brack_ex_r.to_owned();
    let identity = Mat::with_dims(brack_ex.nrows(), brack_ex.ncols(), |i, j| if i == j { 0.5 } else { 0.0 }); //generate identity matrix for use in wedge computation
    let wedge_prod_fin : MatMut<f64>; //store final wedge product matrix as  1/2*(u^Tv - uv^T)


    matmul(wedge_prod_fin, identity.as_ref(), brack_ex.as_ref(), None, 1.0f64, faer_core::Parallelism::Rayon(0));
    return wedge_prod_fin.to_owned(); //returns wedge_product matrix
} 

fn main() {
    println!("Hello, world!");
}
