use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::path::PathBuf;
use structopt::StructOpt;

type Int = i32;
type Mat = ndarray::Array2<Int>;

const MAX_TRIALS: usize = 10;
const TENURE_SCALE: usize = 5;
const DEFAULT_PERTURBATION: Int = 10;

#[derive(StructOpt)]
struct Arguments {
    traffic_file: PathBuf,
    max_error: f64,
    ternure_len: Option<Int>,
    momentum: Option<f64>,
    max_noise: Option<Int>,
    perturbation: Option<Int>,
}

fn usize_below(th: usize) -> usize {
    fastrand::usize(0..th)
}

#[derive(Debug)]
struct TabuList {
    row_tabu: Vec<Int>,
    col_tabu: Vec<Int>,
    tenure: Int,
}

impl TabuList {
    fn new(size: usize, tenure: Int) -> Self {
        let col_tabu = (0..size).map(|_| 0).collect();
        let row_tabu = (0..size).map(|_| 0).collect();
        Self {
            col_tabu,
            row_tabu,
            tenure,
        }
    }

    fn is_locked(&self, idx: (usize, usize)) -> bool {
        let (i, j) = idx;
        self.row_tabu[i] > 0 && self.row_tabu[j] > 0
    }

    fn next_step(&mut self) {
        self.row_tabu
            .par_iter_mut()
            .filter(|v| **v > 0)
            .for_each(|v| *v -= 1);
        self.col_tabu
            .par_iter_mut()
            .filter(|v| **v > 0)
            .for_each(|v| *v -= 1);
    }

    fn set_tabu(&mut self, idx: (usize, usize)) {
        let (i, j) = idx;
        self.col_tabu[j] = self.tenure;
        self.row_tabu[i] = self.tenure;
    }
}

fn get_sum(mat: &'_ Mat) -> impl IndexedParallelIterator<Item = Int> + '_ {
    let row_iter = mat
        .axis_iter(ndarray::Axis(0))
        .into_par_iter()
        .map(|row| row.sum());
    let col_iter = mat
        .axis_iter(ndarray::Axis(1))
        .into_par_iter()
        .map(|row| row.sum());
    col_iter.zip(row_iter).map(|(c, r)| c + r)
}

fn get_percentage_error<T: Into<f64>>(a: T, b: T) -> f64 {
    let a = a.into();
    let b = b.into();
    (a - b) / a
}

fn get_error<'a>(
    mat: &'a Mat,
    correct: &'a [Int],
) -> impl ParallelIterator<Item = (usize, (f64, Int))> + 'a {
    let row_sum = get_sum(mat);
    correct
        .par_iter()
        .zip(row_sum)
        .map(|(a, b)| (get_percentage_error(*a, b), a - b))
        .enumerate()
}

fn outside_bound(mat: &Mat, traffic_sum: &[Int], max_val: f64) -> Option<Vec<(usize, Int)>> {
    let perc_errors = get_error(mat, traffic_sum);
    let mut tmp: Vec<(usize, f64, Int)> = perc_errors
        .map(|(i, (e, a))| (i, e.abs(), a))
        .filter(|(_, e, _)| *e > max_val)
        .collect();
    if !tmp.is_empty() {
        fastrand::shuffle(&mut tmp);
        Some(tmp.into_iter().map(|(i, _, t)| (i, t)).collect())
    } else {
        None
    }
}

struct IndexTrial {
    count: usize,
    index: usize,
    size: usize,
}

impl IndexTrial {
    fn new(count: usize, index: usize, size: usize) -> Self {
        Self { count, index, size }
    }

    fn next_trial(&mut self) -> Option<usize> {
        if self.count == 0 {
            None
        } else {
            loop {
                let j = fastrand::usize(0..self.size);
                if self.index != j {
                    self.count -= 1;
                    return Some(j);
                }
            }
        }
    }
}

fn find_next_index(
    tabu: &mut TabuList,
    idx_list: &[(usize, Int)],
    size: usize,
) -> Option<((usize, usize), Int)> {
    for (i, t) in idx_list {
        let mut trial = IndexTrial::new(MAX_TRIALS, *i, size);
        while let Some(j) = trial.next_trial() {
            if !tabu.is_locked((*i, j)) {
                tabu.set_tabu((*i, j));
                return Some(((*i, j), *t));
            }
        }
    }
    None
}

fn scale_value(val: Int, count: usize) -> Int {
    val / (count as Int)
}

fn apply_perturbation(update: Int, delta: Int, value: Int) -> Int {
    let perturbation = update + fastrand::i32(-delta..delta);
    if value + perturbation < 0 {
        update
    } else {
        perturbation
    }
}

fn update_traffic_value(output: &mut Mat, index: (usize, usize), abs_err: Int, perturbation: Int) {
    let (i, j) = index;
    let update = scale_value(abs_err, output.nrows());
    let tmp = output[(i, j)] + update;
    if tmp > 0 {
        output[(i, j)] += apply_perturbation(update, perturbation, output[(i, j)]);
        output[(j, i)] += apply_perturbation(update, perturbation, output[(j, i)]);
    } else {
        output[(i, j)] = 1;
        output[(j, i)] = 1;
    }
}

fn random_index(size: usize) -> (usize, usize) {
    (usize_below(size), usize_below(size))
}

fn potential_noise_index(mat: &Mat) -> Option<(usize, usize)> {
    let size = mat.ncols();
    for _ in 0..MAX_TRIALS {
        let idx = random_index(size);
        if mat[idx] == 1 {
            return Some(idx);
        }
    }
    None
}

fn noise_momentum(output: &mut Mat, momentum: f64, max_val: Int) {
    let p = fastrand::f64();
    if p < momentum {
        let val = fastrand::i32(0..max_val);
        if let Some((i, j)) = potential_noise_index(output) {
            output[(i, j)] = val;
            output[(j, i)] = val;
        }
    }
}

fn rebuild_arc_traffic(config: &Config) -> Mat {
    let size = config.traffic_vector.len();
    let mut output = ndarray::Array2::zeros((size, size));
    let mut tabu = TabuList::new(size, config.tenure);
    while let Some(error) = outside_bound(&output, &config.traffic_vector, config.max_err) {
        if let Some((idx, et)) = find_next_index(&mut tabu, &error, size) {
            update_traffic_value(&mut output, idx, et, config.perturbation);
        }
        tabu.next_step();
        noise_momentum(&mut output, config.momentum, config.max_noise);
    }
    output
}

struct Config {
    traffic_vector: Vec<Int>,
    max_err: f64,
    tenure: Int,
    momentum: f64,
    max_noise: Int,
    perturbation: Int,
}

impl Config {
    fn from_args(args: Arguments) -> Result<Self, Box<dyn std::error::Error>> {
        let traffic_vector = load_traffic_vector(args.traffic_file)?;
        let tenure = args
            .ternure_len
            .unwrap_or_else(|| get_default_tenure(&traffic_vector));
        let momentum = args.momentum.unwrap_or(0.0);
        let max_noise = args.max_noise.unwrap_or(0);
        let perturbation = args.perturbation.unwrap_or(DEFAULT_PERTURBATION);
        Ok(Self {
            traffic_vector,
            max_err: args.max_error,
            tenure,
            momentum,
            max_noise,
            perturbation,
        })
    }
}

fn get_default_tenure<T>(tv: &[T]) -> Int {
    let len = tv.len();
    (len / TENURE_SCALE) as Int
}

#[derive(Deserialize)]
struct TrafficVector(Vec<Int>);

fn load_traffic_vector(file_path: PathBuf) -> Result<Vec<Int>, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let traffic_vector: TrafficVector = serde_json::from_reader(file)?;
    Ok(traffic_vector.0)
}

fn run_tabu_search(config: Config) {
    let mat = rebuild_arc_traffic(&config);

    for row in mat.rows() {
        for v in row {
            print!("{:04} ", v);
        }
        println!();
    }
    get_sum(&mat)
        .zip(&config.traffic_vector)
        .enumerate()
        .for_each(|val| println!("{:?}", val));
}
fn main() {
    let args = Arguments::from_args();

    match Config::from_args(args) {
        Ok(config) => run_tabu_search(config),
        Err(err) => eprintln!("Error: {:?}", err),
    }
}
