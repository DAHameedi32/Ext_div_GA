use faer_core::Mat;
use rand::Rng;

mod fitness;
mod reproduction;

pub fn GA_main(
    k_forms: Vec<Mat<f64>>,
    k_plus_one_forms: Vec<Mat<f64>>,
    _generations: u8,
    _pop_size: usize,
) -> (Mat<f64>, f64) {
    let mut population = reproduction::pop_init(_pop_size, &k_plus_one_forms[0]); //initialises population of exterior derivative matrices

    //repeat this block of code ad 100 generations
    for _r in 0.._generations {
        let pop_buffer = population.clone();
        let moved_k_forms = k_forms.clone();
        let moved_k_plus_one_forms = k_plus_one_forms.clone();

        let ranked_pop = fitness::rank_pop(&moved_k_forms, &moved_k_plus_one_forms, &pop_buffer); // current error here
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
            new_gen.push(reproduction::reproduce(&breeding_pairs[i].0, &breeding_pairs[i].1));
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
                let bit_string = reproduction::to_bitstring(&new_gen[i]);
                let new_bit_string = reproduction::mutate(bit_string, prob_mut);
                let new_mat = reproduction::mat_from_bitstring(new_bit_string, matrix_buff);
                new_gen[i] = new_mat;
            }
        }
        //let population = new_gen:
        population = new_gen;
        //repeat for 100 generations now
    }
    //finally extract the fittest matrix and return it:
    let final_pop = fitness::rank_pop(&k_forms, &k_plus_one_forms, &population);
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
