#![forbid(unsafe_code)]
extern crate chrono;
extern crate serde;
extern crate serde_json;
pub mod features;
pub mod generators;
pub mod models;
pub mod neural;
pub mod utils;

use chrono::{DateTime, FixedOffset};
use generators::Generator;
use models::{Club, Clubs, DataEntry, Match};
use neural::nn::{HaltCondition, NN};
use std::{
    collections::{HashMap, HashSet},
    fmt,
};
use utils::{normalize};

#[derive(Clone, Debug)]
pub struct Guru<'a> {
    data_set: &'a [Match],
}
#[derive(Debug)]
pub struct NetworkStats {
    pub tested: usize,
    pub positive: usize,
    pub negative: usize,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Stats {
    pub home_scores: Vec<u8>,
    pub away_scores: Vec<u8>,
    pub games_played: [u8; 2],
}

// A Prediction contains the results of a match predicted by the network.
#[derive(Debug)]
pub struct Prediction {
    date: DateTime<FixedOffset>,
    teams: (String, String),
    expected_scores: (u8, u8),
    predicted_scores: (u8, u8),
}

// A wrapper to store a vector of Prediction structs.
#[derive(Debug)]
pub struct Predictions(Vec<Prediction>);

pub trait Markdown {
    fn to_table(&self) -> String;
}
pub trait Testing {
    fn test(
        &self,
        net: &mut NN,
        test_set: &[DataEntry], //&[(Vec<f64>, Vec<f64>)],
        matches: &[Match],
    ) -> ([NetworkStats; 2], Predictions);
}
pub trait Training {
    fn train(
        &self,
        net: &mut NN,
        training_set: &[DataEntry],
        momentum: f64,
        rate: f64,
        halt_error: f64,
    );
}

impl From<&[Match]> for Clubs {
    fn from(matches: &[Match]) -> Self {
        let mut tmp_clubs = HashSet::new();
        for m in matches {
            tmp_clubs.insert(Club::new(m.home.clone()));
            tmp_clubs.insert(Club::new(m.away.clone()));
        }
        let data: HashMap<Club, u32> = tmp_clubs.into_iter().enumerate()
            .map(|(i, club)| {
                (club, i as u32)
            })
            .collect();
        Clubs { data }
    }
}

/// the u8 is the max value used to set the upper limit for a normalization function
impl<T: Generator> From<(&Match, &Clubs, u8, &mut T)> for DataEntry {
    fn from(from: (&Match, &Clubs, u8, &mut T)) -> Self {
        let inputs = from.3.generate(from.0);
        let outputs = if let Some(result) = from.0.result {
            vec![
                normalize(f64::from(result[0]), 0f64, from.2.into()),
                normalize(f64::from(result[1]), 0f64, from.2.into()),
            ]
        } else {
            vec![]
        };
        DataEntry { inputs, outputs }
    }
}

impl<'a> Guru<'a> {
    pub fn new(data_set: &'a [Match]) -> Self {
        Guru { data_set }
    }
}
    /*** Returns the goal difference between
    goals shot at home for the home team at home
    and the away team away
    Takes all played matches into accout
    Example:
        Home Team played 4 matches at home, score 3, 0, 2, 1 = 6
        Away Team played 2 matches away, scored 1, 1 = 2
        Goal difference 3.0
    **/
    // TODO: goal_diff to date
    fn goal_diff(stats: &mut Stats) -> f64 {
        let h = stats
            .home_scores
            .drain(0..stats.games_played[0] as usize)
            .collect::<Vec<u8>>()
            .iter()
            .sum::<u8>();
        let a = stats
            .away_scores
            .drain(0..stats.games_played[1] as usize)
            .collect::<Vec<u8>>()
            .iter()
            .sum::<u8>();
        if a != 0 {
            f64::from(h / a)
        } else {
            f64::from(h)
        }
    }

impl<'a> Testing for Guru<'a> {
    // TODO separate Display
    fn test(
        &self,
        net: &mut NN,
        test_set: &[DataEntry], //  &[(Vec<f64>, Vec<f64>)]
        matches: &[Match],
    ) -> ([NetworkStats; 2], Predictions) {
        let ats = Stats::all_time_highest_score_in_league(&self.data_set);
        let highest = ats.iter().max().unwrap();
        // TODO: move to caller
        let mut res_stats = NetworkStats::default();
        let mut win_stats = NetworkStats::default();
        let test_data: Vec<(Vec<f64>, Vec<f64>)> = test_set
            .iter()
            .map(|e| (e.inputs.clone(), e.outputs.clone()))
            .collect();
        let mut predictions = Predictions(Vec::new());
        for i in 0..test_data.len() {
            let res = net.run(&test_data[i].0);
            let phr = (res[0] * f64::from(*highest).round()) as u8; // denormalized home result
            let par = (res[1] * f64::from(*highest).round()) as u8; // denormalized away result
                                                                    // assuming test else prediction
                                                                    // TODO: move to caller
            if matches[i].result.is_some() {
                // Create a prediction and add it to the Predictions vector.
                let p = Prediction {
                    date: matches[i].date,
                    teams: (matches[i].home.clone(), matches[i].away.clone()),
                    expected_scores: (matches[i].result.unwrap()[0], matches[i].result.unwrap()[1]),
                    predicted_scores: (phr, par),
                };
                predictions.0.push(p);
                // result stats
                if matches[i].result.unwrap() == [phr, par] {
                    res_stats.update(true);
                } else {
                    res_stats.update(false);
                }
                // winner stats
                if matches[i].result.unwrap()[0] > matches[i].result.unwrap()[1] && phr > par
                    || matches[i].result.unwrap()[0] < matches[i].result.unwrap()[1] && phr < par
                    || matches[i].result.unwrap()[0] == matches[i].result.unwrap()[1] && phr == par
                {
                    win_stats.update(true);
                } else {
                    win_stats.update(false);
                }
            } else {
                let p = Prediction {
                    date: matches[i].date,
                    teams: (matches[i].home.clone(), matches[i].away.clone()),
                    expected_scores: (0, 0),
                    predicted_scores: (phr, par),
                };
                predictions.0.push(p);
            }
        }
        ([res_stats, win_stats], predictions)
    }
}

impl<'a> Training for Guru<'a> {
    fn train(
        &self,
        net: &mut NN,
        training_set: &[DataEntry],
        momentum: f64,
        rate: f64,
        halt_error: f64,
    ) {
        if momentum > 1.0 || rate > 1.0 {
            panic!("invoking train(): Values for momentum and rate must be <= 1.0")
        }
        // impl Into for DataEntry
        let test_data: Vec<(Vec<f64>, Vec<f64>)> = training_set
            .iter()
            .map(|e| (e.inputs.clone(), e.outputs.clone()))
            .collect();
        net.train(&test_data)
            .halt_condition(HaltCondition::MSE(halt_error))
            .log_interval(Some(1000))
            .momentum(momentum)
            .rate(rate)
            .go();
    }
}

/// Holds Training Results for the Network
impl NetworkStats {
    pub fn new(tested: usize, positive: usize, negative: usize) -> Self {
        NetworkStats {
            tested,
            positive,
            negative,
        }
    }
    pub fn update(&mut self, positive: bool) {
        self.tested += 1;
        if positive {
            self.positive += 1
        } else {
            self.negative += 1
        }
    }
}

impl Default for NetworkStats {
    fn default() -> Self {
        NetworkStats::new(0, 0, 0)
    }
}

impl fmt::Display for NetworkStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let div = if self.tested == 0 { 1 } else { self.tested };
        write!(
            f,
            "Network Stats
-------------- 
tested: {}, positive: {}, negative: {}, correct: {}%",
            self.tested,
            self.positive,
            self.negative,
            self.positive * 100 / div
        )
    }
}

// A prediction Displays as a single row of a markdown table.
impl fmt::Display for Prediction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(
            f,
            "{} {} : {} {}",
            self.teams.0, self.predicted_scores.0, self.predicted_scores.1, self.teams.1
        )?;
        writeln!(
            f,
            "Expected: {} : {}",
            self.expected_scores.0, self.expected_scores.1
        )?;
        fmt::Result::Ok(())
    }
}

impl Markdown for Prediction {
    fn to_table(&self) -> String {
        format!(
            "|{}|{} : {}|{}|",
            self.teams.0, self.predicted_scores.0, self.predicted_scores.1, self.teams.1
        )
    }
}

// A set of predictions display as a markdown table.
impl fmt::Display for Predictions {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Home | Predicted result | Away")?;
        for elem in self.0.iter() {
            write!(f, "{}", elem)?;
        }
        fmt::Result::Ok(())
    }
}
impl Markdown for Predictions {
    fn to_table(&self) -> String {
        let mut s = String::new();
        s.push_str("|Home|Predicted result|Away|\n");
        s.push_str("|-:|:-:|:-|\n");
        for elem in self.0.iter() {
            s.push_str(&elem.to_table());
            s.push_str("\n");
        }
        s
    }
}

impl Stats {
    /// Returns highest scoring in the league for at home and away
    /// (all time highest scoring at home, all time highes scoring away)
    pub fn all_time_highest_score_in_league(matches: &[Match]) -> [u8; 2] {
        let mut score: [u8; 2] = [0, 0];
        for m in matches {
            if let Some(result) = m.result {
                if result[0] > score[0] {
                    score[0] = result[0];
                } else if result[1] > score[1] {
                    score[1] = result[1];
                }
            }
        }
        score
    }

    /// Returns the number of game days in a Vec<Matches>
    pub fn game_days(matches: &[Match]) -> usize {
        matches.iter().fold(0, |i, _m| i + 1)
    }

    pub fn highest_scoring_in_league_to_date(
        matches: &[Match],
        d: &DateTime<FixedOffset>,
    ) -> [u8; 2] {
        let mut hs: [u8; 2] = [0, 0];
        for n in matches {
            if let Some(result) = n.result {
                if n.date < *d {
                    if result[0] > hs[0] {
                        hs[0] = result[0]
                    }
                    if result[1] > hs[1] {
                        hs[1] = result[1]
                    }
                }
            }
        }
        hs
    }

    /// Returns the alltime highest scoring of a club home or away
    /// (highest scoring for club at hone, highest scoring for club away)
    pub fn highest_scoring_by_club_to_date(stats: &Stats) -> [u8; 2] {
        let mut home: u8 = 0;
        let mut away: u8 = 0;
        for &score in &stats.home_scores {
            if score > home {
                home = score;
            }
        }
        for &score in &stats.away_scores {
            if score > away {
                away = score;
            }
        }
        [home, away]
    }

    // pub fn median_goals_to_date(matches: &[Match], date: &DateTime<FixedOffset>) {
    //     // matches.iter()
    //     //     .filter(|n|
    //     //         &n.date < date &&
    //     //         n.result.is_some()
    //     //     ).fold(&[Vec::new::<u8>(); 2], |g, n| {
    //     //         g
    //     //     }
    //     //     );
    // }
    /// Sums and returns the scoring for home or away matches for given
    pub fn total_scoring_by_club_to_date(stats: &Stats) -> [u8; 2] {
        [
            stats.home_scores.iter().sum::<u8>(),
            stats.away_scores.iter().sum::<u8>(),
        ]
    }

    pub fn update(&mut self, new: Stats) {  *self = new; }
}

impl Default for Stats {
    fn default() -> Self {
        Stats {
            home_scores: vec![],
            away_scores: vec![],
            games_played: [0, 0],
        }
    }
}
