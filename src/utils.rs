use faer_core::{mul::matmul, Mat};

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

//define a wedge product:
//vectors as a matrix: u, v.
//Wedge product is defined as: (u^Tv - uv^T) for row vectors Ref: https://www.math.purdue.edu/~arapura/preprints/diffforms.pdf
//taken as given that u and v have same dimension
//however this only works for vectors, so could only generate 2-forms and maybe 3-forms
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
