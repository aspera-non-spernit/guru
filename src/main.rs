extern crate chrono;
extern crate nn;
extern crate guru;

use chrono::{ DateTime, Local, Utc };


use guru::{ Clubs, ClubName, Features, Guru, neural::nn::NN, Match, Scoring, Stats, Training, Testing };
use std::{ collections::HashMap, convert::TryInto, str::FromStr };


fn matches() -> Vec<Match> {
    vec![
        // NISA MATCHES begin at fourth week 
        Match::new(DateTime::parse_from_rfc3339("2019-08-31T19:30:00-04:00").unwrap(), ClubName::Detroit, ClubName::Philadelphia, Some( (1, 0) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-08-31T22:00:00-04:00").unwrap(), ClubName::Oakland, ClubName::California, Some( (3, 3) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-07T22:00:00-04:00").unwrap(), ClubName::LosAngeles, ClubName::SanDiego, Some( (2, 0) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-14T22:04:00-04:00").unwrap(), ClubName::SanDiego, ClubName::California, Some( (3, 1) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-15T19:00:00-04:00").unwrap(), ClubName::Atlanta, ClubName::Stumptown, Some( (1, 3) ) ),
        Match::new(DateTime::parse_from_rfc3339("2019-09-15T19:00:00-04:00").unwrap(), ClubName::Miami, ClubName::Philadelphia, Some( (8, 1) ) ),
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

fn stats(clubs: &Clubs) -> HashMap<ClubName, Stats> {
    let mut leaguge_stats = HashMap::new();
    for c in &clubs.data {
        leaguge_stats.insert( c.0.name.clone(), Stats::gen_stats(&c.0.name, &matches() ) );
    }
    leaguge_stats
}
/// Creates a Vec of tuples.
/// Each tuple represents a training set, where inputs are at 0
/// and outputs at 1
pub fn sets(set_matches: &Vec<Match>, clubs: &Clubs, league_stats: &mut HashMap<ClubName, Stats>) -> Vec<(Vec<f64>, Vec<f64>)> {
    // Vector of two tuples
    let mut set = vec![];

    for m in &mut set_matches.iter() {
        // Adding 14 Features
        // Clubs. 
        let mut inputs = Guru::club_features(&clubs, &m);

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
        inputs.push( guru::normalize(hs[0] as f64, 0f64, (hs[0] + hs[1]) as f64) );
        inputs.push( guru::normalize(hs[1] as f64, 0f64, (hs[0] + hs[1]) as f64) );

        /**
        Adding 2 features
        Calculates the relative strenghs of a team at home or away
        Normalizes both
        **/
        let h_ths: f64 = Stats::total_scoring_by_club_to_date(&league_stats.get(&m.home).unwrap(), Scoring::Home).into();
        let h_tas: f64 = Stats::total_scoring_by_club_to_date(&league_stats.get(&m.home).unwrap(), Scoring::Home).into();
        let a_ths: f64 = Stats::total_scoring_by_club_to_date(&league_stats.get(&m.away).unwrap(), Scoring::Away).into();
        let a_tas: f64 = Stats::total_scoring_by_club_to_date(&league_stats.get(&m.away).unwrap(), Scoring::Away).into();
        // println!("home rel {:?}", (h_ths / h_tas) as f64);
        // println!("away rel {:?}", (a_ths / a_tas) as f64);

        // LEAGUE STATS

        /***  
        Adding 2 features
        Adds the highest scoring of home team at home and away team away to date
        as relative strength to highest scoring of the league
        **/
        let ats = Stats::all_time_highest_score_in_league(&matches());
        let highest = if ats[0] > ats[1] { ats[0] as f64} else { ats[1] as f64 };
        inputs.push( guru::normalize(hs[0] as f64, 0f64, highest as f64) );
        inputs.push( guru::normalize(hs[1] as f64, 0f64, highest as f64) );

        /***
        Adding 1 feature 1 
        Game Day. The date of the match relative to the schedule (all matches)
        as normalized value
        Most recent match = 1
        earlisest match = 0
        **/
        inputs.push(Guru::game_day(&m.date, &matches()));
        
        // OUTPUTS
        match m.result {
            Some(result) => {
                set.push(
                    (
                        inputs,
                        if highest as f64 != 0.0 { 
                            vec![
                                guru::normalize(result.0 as f64, 0f64, highest),
                                guru::normalize(result.1 as f64, 0f64, highest)
                            ]
                        } else {
                            vec![
                                result.0 as f64,    
                                result.1 as f64
                            ]
                        }
                    )
                );
            },
            None => {
                //println!("No results found in Match. Adding results: [0.0 ,0.0]");
                set.push(
                    (
                        inputs,
                        vec![0f64, 0f64]
                    )
                );
            }
        }               
        //Keep track of games played so the avg_score is based on previous matches and not the overall club avg.
        if let Some(s) = league_stats.get_mut(&m.home) {
            s.games_played = [s.games_played[0] + 1, s.games_played[1]];  
        };
        if let Some(s) = league_stats.get_mut(&m.away) {
            s.games_played = [s.games_played[0], s.games_played[1] + 1];
        };
    }
    set
}

fn main()-> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2{ panic!("ERROR: Need max error rate for training.")}
    // all matches
    let all_matches = matches();
    let clubs = Clubs::from(&all_matches);
    let mut stats = stats(&clubs);

    // PREPPING TRAINING & TEST SETS 
    // Only matches that have Some(result) as output
    let mut training_matches: Vec<Match> = all_matches.iter()
        .filter( |&m| m.result.is_some() )
        .map(|&m| m)
        .collect();
    let test_matches = &training_matches.drain(20..36).collect();
    let training_set = sets(&training_matches, &clubs, &mut stats);  
    let test_set = sets(&test_matches, &clubs, &mut stats);

    // CREATING NETWORK
    let input_nodes: u32 = *(&training_set[0].0.len().try_into().unwrap());
    let mut net = NN::new(&[input_nodes, 19, 2]);
    // TRAIN NETWORK
    Guru::train(&mut net, &training_set, 0.3, 0.2, f64::from_str(&args[1]).unwrap());
    println!();

    // TEST NETWORK
    Guru::test("Testing on TRAINING set", &mut net, &training_set, &training_matches);
    println!();
    println!();
    Guru::test("Testing on TEST set", &mut net, &test_set, &test_matches);
    println!();
    println!();
    // PREDICTION 
    let prediction_matches: Vec<Match> = all_matches.iter()
        .filter( |&m| m.result.is_none() )
        .map(|&m| m)
        .collect();
    prediction_matches.to_vec().sort_by(|a, b| b.date.cmp(&a.date));
    let prediction_set = sets(&prediction_matches, &clubs, &mut stats);
    Guru::test("PREDICTION", &mut net, &prediction_set, &prediction_matches);
    Ok(())
}