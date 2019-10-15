use crate::models::Match;
use std::{fs::File, io::prelude::*};

pub fn load_matches() -> std::io::Result<Vec<Match>> {
    let mut file = File::open("data.json")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let matches: Vec<Match> = serde_json::from_str(&contents).unwrap();
    Ok(matches)
}

/// Simple normalization function
pub fn normalize(v: f64, min: f64, max: f64) -> f64 { 
    if (max - min) == 0.0 {  (v - min)  } else { (v - min) / (max - min) }
}

