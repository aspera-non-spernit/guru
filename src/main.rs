#![forbid(unsafe_code)]
extern crate guru;

use guru::{
    models::{ Clubs, Match, TrainingEntry },
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

fn stats(clubs: &Clubs) -> HashMap<String, Stats> {
    let mut leaguge_stats = HashMap::new();
    for c in &clubs.data {
        leaguge_stats.insert(c.0.name.clone(), Stats::default());
    }
    leaguge_stats
}

#[derive(Clone, Debug)]
struct MyInputGen<'a> { values: (Vec<Match>, &'a Clubs, HashMap<String, Stats>) }

impl Generator for MyInputGen<'_> {
    // 0 training_matches, 1 &clubs, 2 &stats
    fn generate(&mut self, m: &Match) -> Vec<f64> {
        let mut inputs = vec![];

        // 14 features: clubs
        inputs.extend_from_slice( &Guru::club_features(m, self.values.1) );

        // 1 features: Game Day Factor
        // Ealier matches valued less than more recent matches
        inputs.push(Guru::game_day(&m.date, &self.values.0));

        // 2 features
        // Total Home Scoring for Home Team
        // Total Away Scoring for Away Team
        // Normalized with max = sum(hts, ats);
        // Strength of both teams realtive to each other
        let hts = Stats::total_scoring_by_club_to_date(&self.values.2.get(&m.home).unwrap());
        let ats = Stats::total_scoring_by_club_to_date(&self.values.2.get(&m.away).unwrap());
        inputs.push( guru::utils::normalize(hts[0].into(), 0f64, (hts[0] + ats[1]).into()) );
        inputs.push( guru::utils::normalize(ats[1].into(), 0f64, (hts[0] + ats[1]).into()) );

        // 2 features
        // Total Home Scoring for Home Team
        // Total Away Scoring for Away Team
        // Normalized with max = sum(hts, ats);
        // Strength of both teams realtive to the league performance
        // hcil - highest scoring in leaguge in data set/league
        // TODO: should be to date
        let hs = [
            Stats::highest_scoring_by_club_to_date(&self.values.2.get(&m.home).unwrap())[0],
            Stats::highest_scoring_by_club_to_date(&self.values.2.get(&m.away).unwrap())[1],
        ];
        let lhs = Stats::all_time_highest_score_in_league(&self.values.0);
        inputs.push( guru::utils::normalize(hs[0].into(), 0f64, lhs[0].into()) );
        inputs.push( guru::utils::normalize(hs[1].into(), 0f64, lhs[1].into()) );
    
        // let hs: f64 = if hcil[0] > hcil[1] { hcil[0].into() } else { hcil[1].into() };
        // dbg!(hs);
        // inputs.push( guru::utils::normalize(hts[0].into(), 0f64, hs ) );
        // inputs.push( guru::utils::normalize(ats[1].into(), 0f64, hs ) );
        // if guru::utils::normalize(hts[0].into(), 0f64, hs ) > 1f64 {
        //     println!("match {:?} ", m);
        // }



        // Updating Stats
        let mut h_stats = self.values.2.get_mut(&m.home).unwrap().clone();
        let mut a_stats = self.values.2.get_mut(&m.away).unwrap().clone();

        h_stats.home_scores.push(m.result.unwrap()[0]);
        h_stats.games_played[0] += 1;
        a_stats.away_scores.push(m.result.unwrap()[1]);
        a_stats.games_played[1] += 1;

        self.values.2.remove(&m.home);
        self.values.2.insert(String::from(&m.home), h_stats.clone() );
        self.values.2.remove(&m.away);
        self.values.2.insert(String::from(&m.away), a_stats.clone() );
        
        // 
        assert_eq!(
            self.values.2.get(&m.home).unwrap().home_scores.len(),
            self.values.2.get(&m.home).unwrap().games_played[0] as usize
        );
        assert_eq!(
            self.values.2.get(&m.home).unwrap().away_scores.len(),
            self.values.2.get(&m.home).unwrap().games_played[1] as usize
        );
        assert_eq!(
            self.values.2.get(&m.away).unwrap().home_scores.len(),
            self.values.2.get(&m.away).unwrap().games_played[0] as usize
        );
        assert_eq!(
            self.values.2.get(&m.away).unwrap().away_scores.len(),
            self.values.2.get(&m.away).unwrap().games_played[1] as usize
        );
        dbg!(&inputs);
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
    let mut stats = stats(&clubs);
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
    
    let mut my_in_gen = MyInputGen {
        values: (
            training_matches.clone(),
            &clubs, 
            stats.clone()
        )
    };
    for m in training_matches {
        training_set.push(TrainingEntry::from( (&m, &clubs, max, &mut my_in_gen) ));
    }
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
