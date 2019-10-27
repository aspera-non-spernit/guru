#![forbid(unsafe_code)]
#[macro_use]
extern crate clap;
extern crate guru;

use clap::App;
use guru::{
    generators::DefaultInputGenerator,
    models::{Clubs, DataEntry, Match},
    neural::nn::NN,
    utils::{load_matches, load_network, filter_results, filter_no_results, rand_k_split, save_network},
    Guru, Markdown, Stats, Testing, Training,
};
use std::{collections::HashMap, str::FromStr};

fn stats(clubs: &Clubs) -> HashMap<String, Stats> {
    let mut league_stats = HashMap::new();
    for c in &clubs.data {
        league_stats.insert(c.0.name.clone(), Stats::default());
    }
    league_stats
}

fn main() -> std::io::Result<()> {
    let yaml = load_yaml!("../../config/cli.yml");
    let opts = App::from_yaml(yaml).get_matches();
    let error = f64::from_str(opts.value_of("error").unwrap()).unwrap();
    let all_matches = if let Some(f) = opts.value_of("data") {
        load_matches(f)?
    } else {
        // example matches
        load_matches("data/data.json")?
    };
    // must be sorted, doesn't matter to what,
    // network needs the same order otherwise the nwtwork
    // would not be able to calculate error correctly for networks loaded from file
    // TODO: avoid sorted, collect into all_matches
    let mut sorted: Vec<Match> = all_matches.to_vec();
    sorted.sort_by(|a, b| a.date.cmp(&b.date).to_owned());
    // Clubs is required because ```Club```(s) are taken from a set of matches (data.json) without
    // ids
    let clubs: Clubs = Clubs::from(sorted.as_slice());
    let stats = stats(&clubs);
    let guru = Guru::new(&sorted);

    let mut ttraining_matchesm: Vec<Match> = filter_results(&sorted);

    // taking n% from training_matches for testing.
    let split: f32 = if opts.is_present("split-data") {
        opts.value_of("split-data").unwrap().parse().unwrap()
    } else {
        0.9
    };
    let upper: usize = (training_matches.len() as f32 * split).round() as usize;
    let test_matches: Vec<Match> = training_matches
        .drain(upper..training_matches.len())
        .collect();
    // using matches in the data set that have no result (match in the future) to predict the result
    // for those matches
    let prediction_matches: Vec<Match> = filter_no_results(&sorted);

    // required for normalization of results (output)
    let ats = Stats::all_time_highest_score_in_league(&sorted);
    // TODO: let Generator do that
    let max = if ats[0] > ats[1] { ats[0] } else { ats[1] };
    let mut def_in_gen = DefaultInputGenerator {
        values: (training_matches.clone(), &clubs, stats.clone()),
    };

    let training_set: Vec<DataEntry> = training_matches
        .iter()
        .map(|m| DataEntry::from((m, &clubs, max, &mut def_in_gen)))
        .collect();
    let test_set: Vec<DataEntry> = test_matches
        .iter()
        .map(|m| DataEntry::from((m, &clubs, max, &mut def_in_gen)))
        .collect();
    let prediction_set: Vec<DataEntry> = prediction_matches
        .iter()
        .map(|m| DataEntry::from((m, &clubs, max, &mut def_in_gen)))
        .collect();

    // Creating the network
    //let _hidden_size = (training_set[0].inputs.len() as f64 * 0.66).round() as u32;
    let mut net = if opts.is_present("load-network") {
        load_network()?
    } else {
        NN::new(&[
            training_set[0].inputs.len() as u32,
            12,
            8,
            5,
            training_set[0].outputs.len() as u32,
        ])
    };
    if !opts.is_present("no-train") {
        println!("Training Prediction Network...");
        guru.train(&mut net, &training_set, 0.3, 0.2, error);
    }

    if opts.is_present("save-network") {
        save_network(&net)?;
    }

    // testing / validating
    let (test_results, predictions) = guru.test(&mut net, &training_set, &training_matches);
    println!("Testing on (seen) Training Data");
    println!("{}", predictions);
    println!("Result {}\n", test_results[0].to_string());
    println!("Winner {}", test_results[1].to_string());
    println!("--------------------------\n\n");
    let (test_results, test_predictions) = guru.test(&mut net, &test_set, &test_matches);
    println!("Testing on (unseen) Test Data");
    println!("{}", test_predictions);
    println!("Result {}\n", test_results[0].to_string());
    println!("Winner {}", test_results[1].to_string());
    println!("--------------------------\n\n");
    // predict future matches
    let (_test_results, predictions) = guru.test(&mut net, &prediction_set, &prediction_matches);
    // TODO: Fix empty
    println!("Predicting future matches: \n");
    println!("{}", predictions.to_table());
    Ok(())
}
