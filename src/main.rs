#![forbid(unsafe_code)]
extern crate guru;

use guru::{
    models::{ Clubs, Match, Scoring, TrainingEntry },
    Features,
    Guru, 
    Generator,
    neural::nn::NN, 
    Stats,
    utils::{ load_matches, normalize },
    Training,
    Testing
};
use std::{collections::HashMap, str::FromStr};

fn stats(clubs: &Clubs, matches: &[Match]) -> HashMap<String, Stats> {
    let mut leaguge_stats = HashMap::new();
    for c in &clubs.data {
        leaguge_stats.insert(c.0.name.clone(), Stats::default());
    }
    leaguge_stats
}

#[derive(Clone, Debug)]
struct MyInputGen<'a> { values: (&'a Vec<Match>, &'a Clubs, &'a HashMap<String, Stats>) }

impl Generator for MyInputGen<'_> {
    // 0 training_matches, 1 &clubs, 2 &stats
    fn generate(&self, m: &Match) -> Vec<f64> {
        let mut inputs = vec![];
        inputs.extend_from_slice( &Guru::club_features(m, self.values.1) );
        inputs
    }
}
fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        panic!("ERROR: Need max error rate for training.")
    }
    // all matches
    let all_matches = load_matches()?;
    // Clubs is required because Clubs are taken from a set of matches (data.json)
    // A single club doesn't have an id. 
    let clubs: Clubs = Clubs::from(all_matches.as_slice());
    let mut stats = stats(&clubs, &all_matches);
    let guru = Guru::new(&all_matches);

    let training_matches: Vec<Match> = all_matches
        .iter()
        .filter(|&m| m.result.is_some())
        .cloned()
        .collect();

    // required for normalization of results (output)
    let ats = Stats::all_time_highest_score_in_league(&all_matches);
    let max = if ats[0] > ats[1] { ats[0] } else { ats[1] };
    
    let mut training_set = vec![];
    
    let my_in_gen = MyInputGen {
        values: (
            &training_matches.clone(),
            &clubs, 
            &stats
        )
    };
    for m in training_matches {
        training_set.push(TrainingEntry::from( (&m, &clubs, max, my_in_gen.clone()) ));
    }
    dbg!(&training_set);
    // Creating the network
    let _hidden_size = (training_set[0].inputs.len() as f64 * 0.66).round() as u32;
    let mut pred_net = NN::new(&[
        training_set[0].inputs.len() as u32,
        15,
        9,
        training_set[0].outputs.len() as u32,
    ]);
    println!("Training Prediction Network...");
    // TRAIN NETWORKS
    guru.train(
        &mut pred_net,
        &training_set,
        0.3,
        0.2,
        f64::from_str(&args[1]).unwrap(),
    );

    // TESTING NETWORKS
    // Prediction Network
    // TODO: for each Match
    // let mut test_results = guru.test(
    //     "Testing Prediction Network on Training Set",
    //     &mut pred_net,
    //     &pred_training_set,
    //     &training_matches,
    // );
    // println!("\n\n\n\nResult {}\n", test_results[0].to_string());
    // println!("Winner {}\n\n\n\n", test_results[1].to_string());
    // test_results = guru.test(
    //     "Testing on Test Set",
    //     &mut pred_net,
    //     &pred_test_set,
    //     &test_matches,
    // );
    // println!("Result {}\n", test_results[0].to_string());
    // println!("Winner {}", test_results[1].to_string());
    // println!("--------------------------");

    // // Predict the future and become rich
    // let prediction_matches: Vec<Match> = all_matches
    //     .iter()
    //     .filter(|&m| m.result.is_none())
    //     .cloned()
    //     .collect();
    // // dbg!(&prediction_matches);
    // let prediction_set: Vec<(Vec<f64>, Vec<f64>)> =
    //     input_sets(&prediction_matches, &clubs, &mut stats, &all_matches)
    //         .iter()
    //         .map(|m| (m.clone(), vec![]))
    //         .collect();
    // guru.test(
    //     "Predicting future matches..",
    //     &mut pred_net,
    //     &prediction_set,
    //     &prediction_matches,
    // );
    
    Ok(())
}
