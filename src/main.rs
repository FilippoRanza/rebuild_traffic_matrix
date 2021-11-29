use fastrand;
use ndarray;
use std::cmp::Ordering;

use rayon::prelude::*;

type Int = i32;
type Mat = ndarray::Array2<Int>;

#[derive(Debug)]
struct TabuList {
    tabu: Mat,
    tenure: Int,
}

impl TabuList {
    fn new(size: usize, tenure: Int) -> Self {
        let tabu = ndarray::Array2::zeros((size, size));
        Self { tabu, tenure }
    }

    fn is_locked(&self, idx: (usize, usize)) -> bool {
        self.tabu[idx] > 0
    }

    fn next_step(&mut self) {
        self.tabu
            .par_iter_mut()
            .filter(|v| **v > 0)
            .for_each(|v| *v -= 1);
    }

    fn set_tabu(&mut self, idx: (usize, usize)) {
        let (i, j) = idx;
        self.tabu.row_mut(i).into_par_iter().for_each(|v| *v = self.tenure);
        self.tabu.column_mut(j).into_par_iter().for_each(|v| *v = self.tenure);
    }
}

fn get_sum<'a>(mat: &'a Mat) -> impl Iterator<Item = Int> + 'a {
    mat.rows().into_iter().map(|row| row.sum())
}

fn get_percentage_error<T: Into<f64>>(a: T, b: T) -> f64 {
    let a = a.into();
    let b = b.into();
    (a - b) / a
}

fn get_error<'a>(mat: &'a Mat, correct: &'a Vec<Int>) -> impl Iterator<Item = (usize, f64)> + 'a {
    let row_sum = get_sum(mat);
    correct
        .into_iter()
        .zip(row_sum)
        .map(|(a, b)| get_percentage_error(*a, b))
        .enumerate()
}

fn f64_comp(a: &(usize, f64, ErrorType), b: &(usize, f64, ErrorType)) -> Ordering {
    let a = a.1;
    let b = b.1;
    if a < b {
        Ordering::Less
    } else if b < a {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

#[derive(Clone, Copy, Debug)]
enum ErrorType {
    Positive,
    Negative,
}

impl ErrorType {
    fn from_error(err: (usize, f64)) -> (usize, f64, Self) {
        if err.1 < 0.0 {
            (err.0, err.1.abs(), Self::Negative)
        } else {
            (err.0, err.1.abs(), Self::Positive)
        }
    }
}

fn outside_bound(
    mat: &Mat,
    traffic_sum: &Vec<Int>,
    max_val: f64,
) -> Option<Vec<(usize, ErrorType)>> {
    let perc_errors = get_error(mat, traffic_sum);
    let mut tmp: Vec<(usize, f64, ErrorType)> = perc_errors
        .map(ErrorType::from_error)
        .filter(|(_, e, _)| *e > max_val)
        .collect();
    if tmp.len() > 0 {
        tmp.sort_by(f64_comp);
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
    idx_list: &[(usize, ErrorType)],
    size: usize,
) -> Option<((usize, usize), ErrorType)> {
    for (i, t) in idx_list {
        let mut trial = IndexTrial::new(20, *i, size);
        while let Some(j) = trial.next_trial() {
            if !tabu.is_locked((*i, j)) {
                tabu.set_tabu((*i, j));
                return Some(((*i, j), *t));
            }
        }
    }
    None
}

fn update_traffic_value(output: &mut Mat, index: (usize, usize), et: ErrorType) {
    let (i, j) = index;
    match et {
        ErrorType::Positive => {
            output[(i, j)] += 5;
            output[(j, i)] += 5;
        }
        ErrorType::Negative => {
            if output[(i, j)] > 1 {
                output[(i, j)] -= 1;
                output[(j, i)] -= 1;
            }
        }
    };
}

fn rebuild_arc_traffic(traffic_sum: &Vec<Int>, max_err: f64, tenure: Int) -> Mat {
    let size = traffic_sum.len();
    let mut output = ndarray::Array2::zeros((size, size));
    let mut tabu = TabuList::new(size, tenure);
    while let Some(error) = outside_bound(&output, traffic_sum, max_err) {
        if let Some((idx, et)) = find_next_index(&mut tabu, &error, size) {
            update_traffic_value(&mut output, idx, et)
        }
        tabu.next_step();
    }
    output
}

fn main() {
    let vect = vec![
        1875, 1696, 2259, 2098, 2008, 2775, 440, 598, 820, 19240, 689, 11795, 30270, 27917, 5658,
        1761, 2259, 1059, 3000, 838, 958, 1772, 3836,
    ];
    let mat = rebuild_arc_traffic(&vect, 0.05, 50);

    for row in mat.rows() {
        for v in row {
            print!("{:04} ", v);
        }
        println!();
    }
    for val in get_sum(&mat).zip(&vect) {
        println!("{:?}", val)
    }
}
