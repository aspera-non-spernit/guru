#![forbid(unsafe_code)]
extern crate chrono;
extern crate serde;
extern crate serde_json;
pub mod models;
pub mod neural;
pub mod utils;

use chrono::{DateTime, FixedOffset};
use models::{Club, Clubs, DataEntry, Match};
use neural::nn::{HaltCondition, NN};
use std::{
    collections::{HashMap, HashSet},
    convert::TryInto,
    fmt,
};
use utils::{generators::Generator, normalize};

/// The AWAY_FACTOR was used to denote the strength of Away Teams across the entire data set.
const AWAY_FACTOR: f64 = 1.0;

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

pub trait Features {
    /// Returns a normalized vector in size of the league (ie 14 clubs in league, len 14).
    /// Each club represents a position in the vector. The Value of the Home Team
    /// in the Vector is HOME_FACTOR = 1.0.
    /// The Value of the Away Team is AWAY_FACTOR = 0.7
    /// Clubs not playing in that much = 0.0
    fn club_features(m: &Match, clubs: &Clubs) -> Vec<f64>;
    /// Returns the game day as normalized value in relation to all game days,
    /// where for the normalization min is the first game day of the "season" and max
    /// the last day of the season
    fn game_day(m: &DateTime<FixedOffset>, schedule: &[Match]) -> f64;
    fn goal_diff(h_stats: &mut Stats) -> f64;
}

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

impl<'a> Features for Guru<'a> {
    fn club_features(m: &Match, clubs: &Clubs) -> Vec<f64> {
        // CLUBS: HOME_FACTOR AWAY_FACTOR
        let num_of_clubs: u32 = clubs.data.len().try_into().unwrap();
        let mut inputs = vec![];
        let home_index = clubs.clone().get_index_by_name(m.home.clone());
        let away_index = clubs.clone().get_index_by_name(m.away.clone());
        for i in 0..num_of_clubs {
            if i == home_index {
                inputs.push(normalize(1.0, 0.0, 1.0 + AWAY_FACTOR));
            } else if i == away_index {
                inputs.push(normalize(AWAY_FACTOR, 0.0, 1.0 + AWAY_FACTOR));
            } else {
                inputs.push(0f64);
            }
        }
        inputs
    }

    fn game_day(match_date: &DateTime<FixedOffset>, schedule: &[Match]) -> f64 {
        // let mut gd: Vec<i64> = schedule.iter()
            // .map(|m| m.date.timestamp())
            // .collect();
        // gd.dedup();
        // let min = gd.iter().min().unwrap();
        // let max = gd.iter().max().unwrap();
        // normalize(match_date.timestamp() as f64, *min as f64, *max as f64)

        let mm: HashSet<i64> = schedule.iter()
            .map(|s| s.date.timestamp())
            .collect();
        normalize(
            match_date.timestamp() as f64,
            *mm.iter().min().unwrap() as f64,
            *mm.iter().max().unwrap() as f64
        )
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
    pub fn update(&mut self, new: Stats) {  *self = new; }

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

    /***
    Returns the either the wins, draws or losses to date for the home team at home
    and the away team away.
    Note:
        Ordering::Greater is for Wins
        Ordering::Less for Losses
        Ordering::Equal for Draws
    
    **/
    pub fn wdl_to_date(matches: &[Match], m: &Match, ord: std::cmp::Ordering) -> [usize; 2] {
        let h = matches.iter()
            .filter(|n|
                n.home == m.home &&
                n.date < m.date &&
                n.result.is_some()
            )     
            .map(|n| n.result.unwrap() )
            .map(|r| r[0].cmp(&r[1]) )
            .filter(|o| o.eq(&ord) )
            .count();
        let a = matches.iter()
            .filter(|n|
                n.away == m.away && 
                n.date < m.date &&
                n.result.is_some()
            )     
            .map(|n| n.result.unwrap() )
            .map(|r| r[0].cmp(&r[1]) )
            .filter(|o| o.eq(&ord) )
            .count();
        [h, a]
    }

    /// Returns the number of game days in a Vec<Matches>
    pub fn game_days(matches: &[Match]) -> usize {
        matches.iter().fold(0, |i, _m| i + 1)
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

    /// Sums and returns the scoring for home or away matches for given
    pub fn total_scoring_by_club_to_date(stats: &Stats) -> [u8; 2] {
        [
            stats.home_scores.iter().sum::<u8>(),
            stats.away_scores.iter().sum::<u8>(),
        ]
    }
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
