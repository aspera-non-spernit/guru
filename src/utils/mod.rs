use crate::{models::{Match, Sets}, neural::nn::NN};
use rand::prelude::*;
use std::{fs::File, io::prelude::*};

/// A collection of utility functions.

/**
Loads training data from a file.

The file is a text file representing a set of matches in the json format.
The parent is an array. The children are deserializable ```struct Match```.
The field date is a RFC-3399 encoded Date with mandatory timezone.

Example:
```rust
[
  {
    "date": "2019-05-12T19:00:00-04:00",
    "home": "Michigan Stars FC",
    "away": "Detroit City FC",
    "result": [
      0,
      1
    ]
  },
  .. // more matches
]
```
**/
pub fn load_matches(match_file: &str) -> std::io::Result<Vec<Match>> {
    println!("loading data from: {:?}", &match_file);
    let mut file = File::open(match_file)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let matches: Vec<Match> = serde_json::from_str(&contents).unwrap();
    Ok(matches)
}

/// Loads a trained network to file
pub fn load_network() -> std::io::Result<NN> {
    println!("loading network..");
    let mut file = File::open("guru.net")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(NN::from_json(&contents))
}

/// Saves a trained network to file
pub fn save_network(nn: &NN) -> std::io::Result<()> {
    println!("saving network..");
    let mut f = File::create("guru.net")?;
    f.write_all(nn.to_json().as_bytes())?;
    f.sync_all()?;
    Ok(())
}
/// Simple normalization function
pub fn normalize(v: f64, min: f64, max: f64) -> f64 {
    if (max - min) == 0.0 {
        (v - min)
    } else {
        (v - min) / (max - min)
    }
}

/**
    Filters from a data set D all matches that have Some(result)
**/
pub fn filter_results(data_set: &[Match]) -> Vec<Match> {
    data_set.iter()
        .filter(|&m| m.result.is_some())
        .cloned()
        .collect()
}
/**
    Filters from a data set D matches where match.result == None
**/
pub fn filter_no_results(data_set: &[Match]) -> Vec<Match> {
    data_set.iter()
      .filter(|&m| m.result.is_none())
      .cloned()
      .collect()
}


/**
  Splits the data set D  &[Match] into k subsets D of equal size.
  Does not check for Some(result), if you want sub sets with Some(result) only
  pass a set with Some(result) only.
  If you want to keep a reference to the original data set D for convenience,
  pass original: true.
  data[last] may have a different size than the previous.
**/
pub fn rand_k_split<'a>(data_set: &'a Vec<Match>, k: usize, original: bool) -> Sets<'a> {
    let mut cloned = &mut data_set.clone();
    let dsl = &cloned.len();
    let mut k_sets: Vec<Vec<Match>> = vec![];
    let mut iter = data_set.iter();
    let mut rng = rand::thread_rng();

    for i in 0..k {
        k_sets.push(vec![]);
        let mut n = 0;
        while k_sets[i].len() != dsl / k {
            n = rng.gen_range(0, cloned.len());
            k_sets[i].push(cloned[n].clone());
            cloned.remove(n);
        }
    }
    if original {
        Sets::new(Some(data_set), k_sets.clone())
    } else {
        Sets::new(None, k_sets.clone())
    }
}