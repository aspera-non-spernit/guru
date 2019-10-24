use crate::{
  models::Match,
  neural::nn::NN
};
use std::{fs::File, io::prelude::*};

pub mod generators;

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
