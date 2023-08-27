use faer_core::{Mat, mul::matmul, MatRef, mat};

#[allow(non_snake_case)]

//Need to define a function to flatten tensor to matrix:
pub fn flatten_to_mat(tensor:Vec<Vec<Mat<f64>>>) -> Mat<f64>
{
   let y_dim_outer = tensor.len();
   let x_dim_outer = tensor[0].len();
   let y_dim_inner = tensor[0][0].ncols();
   let x_dim_inner = tensor[0][0].nrows();

   let fin_dims_y = (y_dim_outer as i64 + y_dim_inner as i64);
   let fin_dims_x = (x_dim_outer as i64 + x_dim_inner as i64);
   let mut fin_mat: Mat<f64> = Mat::zeros(fin_dims_y as usize, fin_dims_x as usize);
   for y_out in 0..y_dim_outer{
        for x_out in 0..x_dim_outer{
            for y_in in 0..y_dim_inner{
                for x_in in 0..x_dim_inner{
                    let current_pos_y = y_out*y_dim_outer + y_in;
                    let current_pos_x = x_out*x_dim_outer + x_in;
                    fin_mat.write(current_pos_y, current_pos_x, tensor[y_out][x_out].read(y_in, x_in));
                }
            }
        }
   }
   return fin_mat;
}

//define a wedge product:
//vectors as a matrix: u, v.
//Wedge product is defined as: 1/2 (u^Tv - uv^T) Ref: https://www.math.purdue.edu/~arapura/preprints/diffforms.pdf 
//taken as given that u and v have same dimension
//however this only works for vectors, so could only generate poincare dual forms
//as such define the new wedge product as (u dp v) - (v dp u)
pub fn direct_prod(A: MatRef<f64>, B: MatRef<f64>) -> Mat<f64>
{
    //initialise matrix A
    let A_dims_cols = A.ncols() as i64;
    let A_dims_rows = A.nrows() as i64;
    let mut A_buf: Mat<f64> = Mat::zeros(A_dims_cols as usize, A_dims_rows as usize);
    for i in 0..A_dims_rows{
        for j in 0..A_dims_cols{
            A_buf.write(i as usize, j as usize, A.read(i as usize, j as usize));
        }
    }
    
    //initialise matrix B
    let B_dims_cols = B.ncols() as i64;
    let B_dims_rows = B.nrows() as i64;
    let mut B_buf: Mat<f64> = Mat::zeros(B_dims_cols as usize, B_dims_rows as usize);
    for i in 0..B_dims_rows{
        for j in 0..B_dims_cols{
            B_buf.write(i as usize, j as usize, B.read(i as usize, j as usize));
        }
    }
    //compute and collect n+1 rank tensor sub matrices in a vector
    let mut tensor_buf:Vec<Mat<f64>> = vec![];
    for i in 0..A_dims_rows{
        for j in 0..A_dims_cols{
            let aux_mat : Mat<f64> = Mat::with_dims(A_dims_rows as usize, A_dims_cols as usize, |x, y| if x == y { A_buf.read(i as usize, j as usize) } else { 0.0 }); 
            let mut targ_mat = Mat::zeros(B_dims_rows as usize, B_dims_cols as usize);
            matmul(targ_mat.as_mut(), aux_mat.as_ref(), B_buf.as_ref(), None, 1.0, faer_core::Parallelism::Rayon(0));
            tensor_buf.push(targ_mat);
        }
    }
    //initialise target matrix C
    let C_cols = (A_dims_cols * B_dims_cols) as usize;
    let C_rows = (A_dims_rows * B_dims_rows) as usize;
    let mut C = Mat::zeros(C_rows, C_cols);
    for i in 0..(tensor_buf.len() as usize){ //iterates through the tensor buffer
        //takes first entry which is a matrix of B_dims * B_dims
        //moves 0, 0 to entry 0, 0
        //moves 0, 1 to entry 0, 1
        //moves 1, 0 to entry 1, 0
        //moves 1, 1 to entry 1, 1 etc until (B_dims, B_dims) is mapped
        //moves to next entry and repeats process for:
        //0,0 to 0+B_dims,0+B_dims
        //0,1 to 0+B_dims,1+B_dims
        //1,0 to 1+B_dims,0+B_dims
        //1, 1 to 1+B_dims,1+B_dims

        for y in 0..B_dims_rows{
            for x in 0..B_dims_cols{
                //general formula moves (x, y) in B in tesnor_buf to (x+(B_dims*i), y+(B_dims*i)) in C
                C.write(
                    (y + (B_dims_rows * i as i64)) as usize,                          /* Row*/
                    (x + (B_dims_cols * i as i64)) as usize,                          /* Column*/
                    tensor_buf[i].read(x as usize, y as usize)             /* Value*/
                ); // maps element (x, y) -> (x+n*i, y+n*i)
            }
        }

    }
    //return matrix C which is direct product as rank 2 tensor rep
    return C;
}


pub fn wedge(u : MatRef<f64>, v : MatRef<f64>) -> Mat<f64>{ //remember to as_ref() the row vectors when you call the function
    //need to fill out matrices for the associated vectors, as they are both n-dimensional row vectors we convert to an nxn matrix with the vector taking up the first row, 0 in all other entries
    let mut u_buf = Mat::zeros(u.ncols(), u.ncols());
    let mut v_buf = Mat::zeros(v.ncols(), v.ncols());

    for i in 0..u_buf.ncols()
    {
        u_buf.write(0, i, u.read(0, i));
    }
    for j in 0..v_buf.ncols()
    {
        v_buf.write(0, j, v.read(0, j));
    }
    //create some buffer matrices to store the product:
    let mut buff_l = Mat::zeros(u_buf.nrows(), v_buf.ncols());
    let mut buff_r = Mat::zeros(u_buf.nrows(), v_buf.ncols());

    matmul(buff_l.as_mut(), u_buf.transpose(), v_buf.as_ref(), None, 1.0f64, faer_core::Parallelism::Rayon(0)); //computes u^Tv sends -> buff_l
    matmul(buff_r.as_mut(), u_buf.as_ref(), v_buf.transpose(), None, 1.0f64, faer_core::Parallelism::Rayon(0)); //computes uv^T sends -> buff_r

    //compute subtraction in brackets:
    let brack_ex = buff_l - buff_r; //stores final bracketed expression: (u^Tv - uv^T)
    let identity = Mat::with_dims(brack_ex.nrows(), brack_ex.ncols(), |i, j| if i == j { 0.5 } else { 0.0 }); //generate identity matrix for use in wedge computation

    let mut wedge_prod_fin = Mat::zeros(u.ncols(), v.ncols());//store final wedge product matrix as  1/2*(u^Tv - uv^T)
    matmul(wedge_prod_fin.as_mut(), identity.as_ref(), brack_ex.as_ref(), None, 1.0f64, faer_core::Parallelism::Rayon(0));

    return wedge_prod_fin //returns wedge_product matrix
} 

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

    let A = wedge(p.as_ref(), q.as_ref());
    let B = wedge(r.as_ref(), s.as_ref());
    let AwedgeB = wedge(A.as_ref(), B.as_ref());
    
    println!("{:?}", A);
    println!("{:?}", B);
    println!("{:?}", AwedgeB);
    /* 
    let c: Mat<f64> = wedge(p.as_ref(), q.as_ref()); //c is a 2-form
    let k: Mat<f64> = wedge(r.as_ref(), c.as_ref()); //k should be a 3-form as wedge product is associative
    let K: Mat<f64> = wedge(s.as_ref(), k.as_ref()); //should be a 4-form
    println!("{:?}", c);
    println!("{:?}", k);
    println!("{:?}", K);
    */
    //we can now learn the exterior derivative, which we will take to be a map from a k-form to a k+1 form:
}
