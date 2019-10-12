use crate::models::Match;
use std::{ fs::File, io::{ BufReader, prelude::* } };

pub fn load_matches<'a>() -> std::io::Result<Vec<Match>> {
    let mut file = File::open("data.json")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let matches: Vec<Match> = serde_json::from_str(&contents).unwrap();
    Ok(matches)
}