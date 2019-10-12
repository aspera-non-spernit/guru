#![forbid(unsafe_code)]
extern crate chrono;
extern crate nn;
extern crate guru;

use chrono::{ DateTime };
use guru::{
    models::{ Clubs, ClubName, Match, Scoring },
    Features,
    Guru, 
    neural::nn::NN, 
    Stats,
    Training,
    Testing
};
use std::{ collections::HashMap, str::FromStr };

#[derive(Debug)] pub struct PredictionResult (f64, f64);
#[derive(Debug)] pub struct ClassificationResult (f64, f64, f64);

/// ResultClassification is a Vector of three unnamed f64 values
/// that can be used store output values for the linear classification pf results
impl From<&Match> for ClassificationResult {
    fn from(m: &Match) -> Self {
        if let Some(result) = m.result {
            if result.0 > result.1 { ClassificationResult(1.0, 0.0, 0.0) }
            else if result.0 == result.1 { ClassificationResult(0.0, 1.0, 0.0) }
            else { ClassificationResult(0.0, 0.0, 1.0) }
        } else {
            panic!("from(m): Match needs Some(result)")
        }
    }
}

/// ResultPrediction is a Vector of two unnamed f64 values
/// that can be used store output values for the training and testing of results
impl From<&Match> for PredictionResult {
    fn from(m: &Match) -> Self {
        let all_matches = matches(); // provided Vec<Match> may not contain all info needed for normalization of values 
        let ats = Stats::all_time_highest_score_in_league(&all_matches);
        let highest = if ats[0] > ats[1] { f64::from(ats[0]) } else { f64::from(ats[1]) };
        match m.result {
            Some(result) => {
                if highest as f64 != 0.0 { 
                    PredictionResult(
                        guru::normalize(f64::from(result.0), 0f64, highest),
                        guru::normalize(f64::from(result.1), 0f64, highest)
                    )
                } else {
                    PredictionResult(
                        f64::from(result.0),
                        f64::from(result.1)
                    )
                }
            },
            None => {
                // ASSUMING PREDICTION
                //println!("No results found in Match. Adding results: [0.0 ,0.0]");
                PredictionResult(0.0, 0.0) 
            }
        }
    }
}

fn stats(clubs: &Clubs) -> HashMap<ClubName, Stats> {
    let mut leaguge_stats = HashMap::new();
    for c in &clubs.data {
        leaguge_stats.insert(c.0.name, Stats::gen_stats(c.0.name, &matches() ) );
    }
    leaguge_stats
}

/// Creates a Vec of tuples.
/// Each tuple represents a training set, where inputs are at 0
/// and outputs at 1
pub fn input_sets<S: ::std::hash::BuildHasher>(set_matches: &[Match], clubs: &Clubs, league_stats: &mut HashMap<ClubName, Stats, S>) -> Vec<Vec<f64>> {
    // vec of input values f64
    // returned
    let mut input_sets = vec![];
    for m in &mut set_matches.iter() {
        let mut inputs = vec![];

        // Adding 14 Features
        // Clubs.
        inputs.extend_from_slice(&Guru::club_features(&clubs, &m));

        /***
        Adding 2 features: Relative strength by total scoring for each team and to date rnage
        Sums home scores for home team (ths) and away scores for away team (tas) to date
        Normalizes with min: 0 max: sum(ths, tas)
        Sum of both features == 1
        **/
        let ths: f64 = Stats::total_scoring_by_club_to_date(&league_stats.get(&m.home).unwrap(), Scoring::Home).into();
        let tas: f64 = Stats::total_scoring_by_club_to_date(&league_stats.get(&m.away).unwrap(), Scoring::Away).into();
        inputs.push(guru::normalize( ths, 0f64, (ths + tas) as f64 ));
        inputs.push(guru::normalize( tas, 0f64, (ths + tas) as f64 ));
 
        /***
        Adding 2 features
        Finds the alltime highest scoring for both home and away team 
        As relative strength between clubs
        Normalizes with min: 0 max: sum(hsh, hsa)  
        **/
        let hsh: f64 = Stats::highest_alltime_scores_by_club(&league_stats.get(&m.home).unwrap()).0.into();
        let hsa: f64 = Stats::highest_alltime_scores_by_club(&league_stats.get(&m.away).unwrap()).1.into(); 
        inputs.push(guru::normalize(hsh, 0f64, (hsh + hsa) as f64) );
        inputs.push(guru::normalize(hsa, 0f64, (hsh + hsa) as f64) );

        let hs = Stats::highest_scores_to_date(
            &league_stats.get(&m.home).unwrap(),
            &league_stats.get(&m.away).unwrap()
        );

        /***
        Adding 2 features
        Finds the highest scoring for both home team at home and away team away to date
        Normalizes with min: 0 max: sum(hsh, hsa)
        **/
        inputs.push( guru::normalize(f64::from(hs[0]), 0f64, f64::from(hs[0] + hs[1]) ) );
        inputs.push( guru::normalize(f64::from(hs[1]), 0f64, f64::from(hs[0] + hs[1]) ) );

        /***
        Adding 2 features
        Calculates the relative strength of a team at home vs away
        Examples:
            The Home team scored to date and at home 7 goals, 6 goals away. Home Strength 1.6667
            The Away team scored to date and at home 4 goals, and 1 goal away. Away Strength 0.25
            The relative normalized strengths between to teams: Home 0.82353 + Away 0.17647 = 1.0
            If teams played a second match home - away flipped and assuming no goals shot in first match:
            The former Home team scored to date and at home 7 goals, 6 goals away. Away Strength 0.8571
            The former Away team scored to date and home 4 goals, and 1 goal away. Home Strength 4.0
            The relative normalized strengths between to teams: former Home 0,1765 + Away 0,8235 = 1.0
        **/
        let h_ths: f64 = Stats::total_scoring_by_club_to_date(&league_stats.get(&m.home).unwrap(), Scoring::Home).into();
        let h_tas: f64 = Stats::total_scoring_by_club_to_date(&league_stats.get(&m.home).unwrap(), Scoring::Away).into();
        let a_ths: f64 = Stats::total_scoring_by_club_to_date(&league_stats.get(&m.away).unwrap(), Scoring::Home).into();
        let a_tas: f64 = Stats::total_scoring_by_club_to_date(&league_stats.get(&m.away).unwrap(), Scoring::Away).into();
        let h_rel = if h_tas != 0f64 { h_ths / h_tas } else { h_ths };
        let a_rel = if a_ths != 0f64 { a_tas / a_ths } else { a_tas };
        
        // println!("h {:?} {:?} {:?}", h_ths, h_tas, h_rel);
        // println!("a {:?} {:?} {:?}", a_ths, a_tas, a_rel);
        //println!("Home {:?} : {:?} Away",  guru::normalize(h_rel as f64, 0f64, (h_rel + a_rel) as f64),  guru::normalize(a_rel as f64, 0f64, (h_rel + a_rel) as f64));
        inputs.push( guru::normalize(h_rel as f64, 0f64, (h_rel + a_rel) as f64) );
        inputs.push( guru::normalize(a_rel as f64, 0f64, (h_rel + a_rel) as f64) );
        

        // h_ths STATS
        /***  
        Adding 2 features
        Adds the highest scoring of home team at home and away team away to date
        as relative strength to highest scoring of the league
        **/
        let ats = Stats::all_time_highest_score_in_league(&matches());
        let highest = if ats[0] > ats[1] { f64::from(ats[0]) } else { f64::from(ats[1]) };
        inputs.push( guru::normalize(f64::from(hs[0]), 0f64, highest as f64) );
        inputs.push( guru::normalize( f64::from(hs[1]), 0f64, highest as f64) );

        /***
        Adding 1 feature 1 
        Game Day. The date of the match relative to the schedule (all matches)
        as normalized value
        Most recent match = 1
        earlisest match = 0
        **/
        inputs.push(Guru::game_day(&m.date, &matches()));
              
        //Keep track of games played so the avg_score is based on previous matches and not the overall club avg.
        if let Some(s) = league_stats.get_mut(&m.home) {
            s.games_played = [s.games_played[0] + 1, s.games_played[1]];  
        };
        if let Some(s) = league_stats.get_mut(&m.away) {
            s.games_played = [s.games_played[0], s.games_played[1] + 1];
        };
        input_sets.push(inputs);
    }
    input_sets
}

fn matches() -> Vec<Match> {
    vec![
        // NISA MATCHES begin at fourth week 
        Match::new(DateTime::parse_from_rfc3339("2019-08-31T19:30:00-04:00").unwrap(), ClubName::Detroit, ClubName::Philadelphia, Some( (1, 0) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-08-31T22:00:00-04:00").unwrap(), ClubName::Oakland, ClubName::California, Some( (3, 3) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-07T22:00:00-04:00").unwrap(), ClubName::LosAngeles, ClubName::SanDiego, Some( (2, 0) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-14T22:04:00-04:00").unwrap(), ClubName::SanDiego, ClubName::California, Some( (3, 1) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-15T19:00:00-04:00").unwrap(), ClubName::Atlanta, ClubName::Stumptown, Some( (1, 3) ) ),
        // Match::new(DateTime::parse_from_rfc3339("2019-09-15T19:00:00-04:00").unwrap(), ClubName::Miami, ClubName::Philadelphia, Some( (8, 1) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-22T19:00:00-04:00").unwrap(), ClubName::California, ClubName::LosAngeles, Some( (3, 0) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-22T19:00:00-04:00").unwrap(), ClubName::Miami, ClubName::Stumptown, Some( (2, 1) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-25T22:00:00-04:00").unwrap(), ClubName::California, ClubName::Oakland, Some( (1, 1) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-28T22:04:00-04:00").unwrap(), ClubName::Stumptown, ClubName::Miami, Some( (0, 2) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-28T22:04:00-04:00").unwrap(), ClubName::SanDiego, ClubName::Oakland, Some( (4, 3) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-10-02T22:00:00-04:00").unwrap(), ClubName::California, ClubName::SanDiego, Some( (3, 0) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-10-05T19:30:00-04:00").unwrap(), ClubName::LosAngeles, ClubName::Oakland, Some( (1, 0) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-10-06T19:30:00-04:00").unwrap(), ClubName::Miami, ClubName::Atlanta, Some( (2, 2) ) ),
        // NEXT PREDICTIONS
        Match::new(DateTime::parse_from_rfc3339("2019-10-12T22:04:00-04:00").unwrap(), ClubName::Atlanta, ClubName::Miami, None ),
        Match::new(DateTime::parse_from_rfc3339("2019-10-12T22:04:00-04:00").unwrap(), ClubName::Stumptown, ClubName::Chattanooga, None ),
        // UPCOMING NISA MATCHES
        Match::new(DateTime::parse_from_rfc3339("2019-10-19T22:04:00-04:00").unwrap(), ClubName::Chattanooga, ClubName::Stumptown, None ),
        Match::new(DateTime::parse_from_rfc3339("2019-10-19T22:04:00-04:00").unwrap(), ClubName::Oakland, ClubName::LosAngeles, None ),
        Match::new(DateTime::parse_from_rfc3339("2019-10-20T22:04:00-04:00").unwrap(), ClubName::California, ClubName::SanDiego, None ),
        Match::new(DateTime::parse_from_rfc3339("2019-10-25T22:04:00-04:00").unwrap(), ClubName::Stumptown, ClubName::Atlanta, None ),
        Match::new(DateTime::parse_from_rfc3339("2019-10-26T22:04:00-04:00").unwrap(), ClubName::Miami, ClubName::Oakland, None ),
        Match::new(DateTime::parse_from_rfc3339("2019-10-26T22:04:00-04:00").unwrap(), ClubName::LosAngeles, ClubName::California, None ),
        Match::new(DateTime::parse_from_rfc3339("2019-11-02T22:04:00-04:00").unwrap(), ClubName::SanDiego, ClubName::LosAngeles, None ),
        // MEMBERS CUP
        Match::new(DateTime::parse_from_rfc3339("2019-08-10T19:30:00-04:00").unwrap(), ClubName::Milwaukee, ClubName::NapaValley, Some( (2, 0) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-08-10T19:30:00-04:00").unwrap(), ClubName::Chattanooga, ClubName::NewYork, Some( (0, 1) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-08-17T19:30:00-04:00").unwrap(), ClubName::NewYork, ClubName::Milwaukee, Some( (2, 2) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-08-17T19:30:00-04:00").unwrap(), ClubName::Detroit,  ClubName::Chattanooga, Some( (2, 1) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-08-24T19:30:00-04:00").unwrap(), ClubName::NapaValley, ClubName::Detroit, Some( (0, 4) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-08-24T19:30:00-04:00").unwrap(), ClubName::Chattanooga, ClubName::Michigan, Some( (1, 1) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-08-31T19:30:00-04:00").unwrap(), ClubName::Michigan, ClubName::NewYork, Some( (0, 2) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-01T19:30:00-04:00").unwrap(), ClubName::NapaValley, ClubName::Chattanooga, Some( (0, 6) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-07T19:30:00-04:00").unwrap(), ClubName::Milwaukee, ClubName::Chattanooga, Some( (1, 1) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-07T19:30:00-04:00").unwrap(), ClubName::NewYork, ClubName::NapaValley, Some( (1, 0) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-14T19:30:00-04:00").unwrap(), ClubName::Chattanooga, ClubName::NapaValley, Some( (3, 0) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-14T19:30:00-04:00").unwrap(), ClubName::NewYork, ClubName::Michigan, Some( (2, 0) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-14T19:30:00-04:00").unwrap(), ClubName::Milwaukee, ClubName::Detroit, Some( (0, 1) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-21T19:30:00-04:00").unwrap(), ClubName::NapaValley, ClubName::Milwaukee, Some( (1, 1) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-21T19:30:00-04:00").unwrap(), ClubName::Michigan, ClubName::Chattanooga, Some( (1, 4) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-21T19:30:00-04:00").unwrap(), ClubName::Detroit, ClubName::NewYork, Some( (1, 1) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-26T19:30:00-04:00").unwrap(), ClubName::Michigan, ClubName::NapaValley, Some( (2, 0) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-28T19:30:00-04:00").unwrap(), ClubName::NewYork, ClubName::Chattanooga, Some( (3, 1) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-28T19:30:00-04:00").unwrap(), ClubName::Michigan, ClubName::Milwaukee, Some( (1, 0) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-28T19:30:00-04:00").unwrap(), ClubName::Detroit, ClubName::NapaValley, Some( (3, 0) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-10-05T19:30:00-04:00").unwrap(), ClubName::Chattanooga, ClubName::Detroit, Some( (0, 1) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-10-05T19:30:00-04:00").unwrap(), ClubName::NewYork, ClubName::Milwaukee, Some( (2, 2) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-10-08T19:30:00-04:00").unwrap(), ClubName::Detroit, ClubName::Michigan, Some( (2, 0) ) ),
        // NEXT PREDICTIONS
        Match::new(DateTime::parse_from_rfc3339("2019-10-12T19:00:00-04:00").unwrap(), ClubName::NewYork, ClubName::Detroit, None ),
        Match::new(DateTime::parse_from_rfc3339("2019-10-12T19:00:00-04:00").unwrap(), ClubName::Milwaukee, ClubName::Michigan, None ),
        // UPCOMING MC MATCHES
        Match::new(DateTime::parse_from_rfc3339("2019-10-16T19:00:00-04:00").unwrap(), ClubName::Michigan, ClubName::Detroit, None ),
        Match::new(DateTime::parse_from_rfc3339("2019-10-16T19:00:00-04:00").unwrap(), ClubName::NapaValley, ClubName::NewYork, None ),
        Match::new(DateTime::parse_from_rfc3339("2019-10-19T19:30:00-04:00").unwrap(), ClubName::Detroit, ClubName::Milwaukee, None ),
        Match::new(DateTime::parse_from_rfc3339("2019-10-26T18:00:00-04:00").unwrap(), ClubName::Chattanooga, ClubName::Milwaukee, None ),
        Match::new(DateTime::parse_from_rfc3339("2019-10-26T19:00:00-04:00").unwrap(), ClubName::NapaValley, ClubName::Michigan, None ),
    ]
}

fn main()-> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2{ panic!("ERROR: Need max error rate for training.")}
    // all matches
    let all_matches = matches();
    let clubs: Clubs = Clubs::from(all_matches.as_slice());
    let mut stats = stats(&clubs);
    let guru = Guru::new(&all_matches);

    // splitting all matches into training and test matches for validation 
    let training_matches: Vec<Match> = all_matches.iter()
        .filter( |&m| m.result.is_some() )
        .copied()
        .collect();
    
    let test_matches: Vec<Match> = training_matches.to_vec().drain(30..training_matches.len()).collect();
    // Create sets for the input nodes
    // Can be used for both networks
    let training_input_sets = input_sets(&training_matches, &clubs, &mut stats);
    let test_input_sets = input_sets(&test_matches, &clubs, &mut stats);

    // OUTPUT SETS
    // let class_result_set: Vec<ClassificationResult> = all_matches.iter()
    //     .filter( |&m| m.result.is_some() )
    //     .map(|m| { ClassificationResult::from(m) } )
    //     .collect();
    let pred_result_set: Vec<PredictionResult> = all_matches.iter()
        .filter( |&m| m.result.is_some() )
        .map(|m| { PredictionResult::from(m) } )
        .collect();
  
    // Zipping input sets and output sets into two training sets (for classification and regression)
    // let class_training_set: Vec< ( Vec<f64>,  Vec<f64>) > = training_input_sets.iter()
    //     .zip( class_result_set.iter() )
    //     .map( |(tis, crs)| (tis.clone(), vec![crs.0, crs.1, crs.2]) )
    //     .collect();
    let pred_training_set: Vec< ( Vec<f64>, Vec<f64>) > = training_input_sets.iter()
        .zip( pred_result_set.iter() )
        .map( |(tis, prs)| (tis.clone(), vec![prs.0, prs.1]) )
        .collect();

    // validation
    // let class_test_set: Vec< ( Vec<f64>, Vec<f64>) > = test_input_sets.iter()
    //     .zip( pred_result_set.iter() )
    //     .map( |(tis, prs)| (tis.clone(), vec![prs.0, prs.1]) )
    //     .collect();
    let pred_test_set: Vec< ( Vec<f64>, Vec<f64>) > = test_input_sets.iter()
        .zip( pred_result_set.iter() )
        .map( |(tis, prs)| (tis.clone(), vec![prs.0, prs.1]) )
        .collect();

    // CREATING NETWORKS
    //let mut class_net = NN::new(&[training_input_sets[0].len() as u32, class_training_set[0].1.len() as u32]);
    // let mut class_net = NN::new(&[
    //         training_input_sets[0].len() as u32,
    //         pred_training_set[0].1.len() as u32
    //     ]
    // );

    let _hidden_size = (training_input_sets[0].len() as f32 * 0.66 ).round() as u32;
    let mut pred_net = NN::new(&[
            training_input_sets[0].len() as u32,
            15, 9,
            pred_training_set[0].1.len() as u32
        ]
    );

    // TRAIN NETWORKS
    //Guru::train("Classification", &mut class_net, &class_training_set.as_slice(), 0.3, 0.2, f64::from_str(&args[1]).unwrap());
    guru.train("Prediction", &mut pred_net, &pred_training_set, 0.3, 0.2, f64::from_str(&args[1]).unwrap());
    println!();

    // TESTING NETWORKS
    // Classification Network
    //guru::class_test("Testing Classification Network on TRAINING set", &mut class_net, &class_training_set, &all_matches);
    println!();
    println!();
    // Prediction Network
    let mut test_results = guru.test("Testing Prediction Network on Training Set", &mut pred_net, &pred_training_set, &training_matches);
    println!();
    println!("Result {}", test_results[0].to_string());
    println!();
    println!("Winner {}", test_results[1].to_string());  
    println!();
    println!();
    println!();
    test_results = guru.test("Testing on Test Set", &mut pred_net, &pred_test_set, &test_matches);
    println!();
    println!("Result {}", test_results[0].to_string());
    println!();
    println!("Winner {}", test_results[1].to_string());  
    println!("--------------------------");
   
    // Predict the future and become rich
    let prediction_matches: Vec<Match> = all_matches.iter()
        .filter( |&m| m.result.is_none() )
        .copied()
        .collect();
   // dbg!(&prediction_matches);
    let prediction_set: Vec<(Vec<f64>, Vec<f64>)> = input_sets(&prediction_matches, &clubs, &mut stats).iter()
        .map(|m| (m.clone(), vec![]) )
        .collect();
    guru.test("Predicting future matches..", &mut pred_net, &prediction_set, &prediction_matches);
    Ok(())
}