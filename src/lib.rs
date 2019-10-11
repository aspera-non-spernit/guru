#[forbid(unsafe_code)]
extern crate chrono;
extern crate serde;
extern crate serde_json;

pub mod neural;

use chrono::{ DateTime, FixedOffset };
use neural::nn::{ NN, HaltCondition };
use std::{ collections::{ HashMap, HashSet }, convert::TryInto, fmt };

const AWAY_FACTOR: f64 = 1.0;

#[derive(Clone, Debug, Eq, Hash, PartialEq)] pub struct Club { pub name: ClubName }

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ClubName { Atlanta, California, Chattanooga, Detroit, LosAngeles, Miami, Michigan, Milwaukee, Oakland, NewYork, NapaValley, Philadelphia, SanDiego, Stumptown }

#[derive(Clone, Debug)] pub struct Clubs { pub data: HashMap<Club, u32> }

#[derive(Clone, Copy, Debug)]
pub struct Match  {
    pub date: chrono::DateTime<FixedOffset>,
    pub home: ClubName,
    pub away: ClubName,
    pub result: Option<(u8, u8)>
}
#[derive(Clone, Debug)] pub struct Guru { data_set: Vec<Match> }
#[derive(Debug)] pub struct NetworkStats {
    tested: usize,
    positive: usize,
    negative: usize
}
#[derive(Debug, Eq, Hash, PartialEq)] pub enum Scoring { Home, Away }
#[derive(Clone, Debug, Eq, Hash, PartialEq)] pub struct Stats { pub home_scores: Vec<u8>, pub away_scores: Vec<u8>, pub games_played: [u8; 2] }

pub trait Features {
    /// Returns a normalized vector in size of the league (ie 14 clubs in league, len 14).
    /// Each club represents a position in the vector. The Value of the Home Team
    /// in the Vector is HOME_FACTOR = 1.0.
    /// The Value of the Away Team is AWAY_FACTOR = 0.7
    /// Clubs not playing in that much = 0.0
    fn club_features(clubs: &Clubs, m: &Match) -> Vec<f64>;
    /// Returns the game day as normalized value in relation to all game days,
    /// where for the normalization min is the first game day of the "season" and max
    /// the last day of the season
    fn game_day(m: &DateTime<FixedOffset>, schedule: &Vec<Match>) -> f64;
    fn goal_diff(h_stats: &mut Stats) -> f64;
}
pub trait Testing { fn test(&self, header: &str, net: &mut NN, test_set: &[(Vec<f64>, Vec<f64>)], matches: &Vec<Match>) -> [NetworkStats; 2]; }
pub trait Training {
    fn train(&self, header: &str, net: &mut NN, training_set: &[(Vec<f64>, Vec<f64>)], momentum: f64, rate: f64, halt_error: f64);
}

impl Club { pub fn new(name: ClubName) -> Self { Club { name } } }

impl fmt::Display for ClubName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Clubs {
    /// Returns the index 
    pub fn get_index(self, club_name: &ClubName) -> u32 {
        let mut i = 0;
        for c in self.data { 
            match c.0.name == *club_name {
                true => { i = c.1.clone(); }
                false => { 0; }
            }
        }
        i
    }
}


impl From<&Vec<Match>> for Clubs {
    fn from(matches: &Vec<Match>) -> Self {
        let mut tmp_clubs = HashSet::new();
        let mut data: HashMap<Club, u32> = HashMap::new();
        for m in matches {
            tmp_clubs.insert(Club::new(m.home));
            tmp_clubs.insert(Club::new(m.away));
        }
        for (i, c) in tmp_clubs.iter().enumerate() {
            data.insert(c.clone(), i as u32);
        }
        Clubs { data }
    }
}

impl Guru {  pub fn new(data_set: Vec<Match>) -> Self { Guru { data_set } } }

impl Features for Guru {
    fn club_features(clubs: &Clubs, m: &Match) -> Vec<f64>  {
        // CLUBS: HOME_FACTOR AWAY_FACTOR
        let num_of_clubs: u32 = clubs.data.len().try_into().unwrap();
        let mut inputs = vec![];
        let home_index = clubs.clone().get_index(&m.home);
        let away_index = clubs.clone().get_index(&m.away);
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

    fn game_day(match_date: &DateTime<FixedOffset>, schedule: &Vec<Match>) -> f64 {
        let mut gd: Vec<i64> = schedule.iter().map(|m| m.date.timestamp() ).collect();
        gd.dedup();
        let min = gd.iter().min().unwrap();
        let max = gd.iter().max().unwrap();
        normalize(match_date.timestamp() as f64, *min as f64, *max as f64)
    }
  
    fn goal_diff(stats: &mut Stats) -> f64 {
        let h = stats.home_scores
            .drain(0..stats.games_played[0] as usize)
            .collect::<Vec<u8>>()
            .iter()
            .sum::<u8>();
        let a = stats.home_scores
            .drain(0..stats.games_played[1] as usize)
            .collect::<Vec<u8>>()
            .iter()
            .sum::<u8>();
        0.0
    }
}
impl Training for Guru {   
    fn train(&self, header: &str, net: &mut NN, training_set: &[(Vec<f64>, Vec<f64>)], momentum: f64, rate: f64, halt_error: f64) {
        println!("Training {:?} Network...", &header);
        if momentum > 1.0 || rate > 1.0 { panic!("invoking train(): Values for momentum and rate must be <= 1.0") }
        net.train(training_set)
            .halt_condition( HaltCondition::MSE(halt_error) )
            .log_interval( Some(1000) )
            .momentum( momentum )
            .rate( rate )
            .go();
    }
}

impl Testing for Guru {
    // TODO separate Display
    fn test(&self, header: &str, net: &mut NN, test_set: &[(Vec<f64>, Vec<f64>)], matches: &Vec<Match>) -> [NetworkStats; 2] { 
        println!("{:?}", header);
        let ats = Stats::all_time_highest_score_in_league(&self.data_set);
        let highest = ats.iter().max().unwrap();
        let mut res_stats = NetworkStats::new();
        let mut win_stats = NetworkStats::new();
        for i in 0..test_set.len() {
            let res = net.run( &test_set[i].0);
            let phr = (res[0] * *highest as f64).round() as u8; // denormalized home result
            let par = (res[1] * *highest as f64).round() as u8; // denormalized away result
            // assuming test else prediction 
            if let Some(result) = matches[i].result {
                println!("{:?} : {:?} on {:?}", matches[i].home, matches[i].away, matches[i].date);
                println!("Expected: {:?} : {:?}", matches[i].result.unwrap().0,  matches[i].result.unwrap().1);
                println!("Predicted:  {:?} : {:?}", phr, par);
                println!();
                // result stats
                if  matches[i].result.unwrap() == (phr,  par) {
                    res_stats.update(true);
                } else {
                    res_stats.update(false);
                }
                // winner stats
                if  matches[i].result.unwrap().0 > matches[i].result.unwrap().1
                    && phr > par
                    || 
                    matches[i].result.unwrap().0 < matches[i].result.unwrap().1
                    && phr < par
                    ||
                    matches[i].result.unwrap().0 == matches[i].result.unwrap().1
                    && phr == par {
                        win_stats.update(true);
                } 
                else {
                    win_stats.update(false);
                }
            } else {
                println!("Prediction: {:?} {:?} : {:?} {:?} on {:?}", matches[i].home, phr, par, matches[i].away, matches[i].date.date());
            }
        }
        [res_stats, win_stats]       
    }
}

impl Match {
    pub fn new(date: DateTime<FixedOffset>, home: ClubName, away: ClubName, result: Option<(u8, u8)>) -> Self {
        Match { date, home, away, result }
    }
}

/// Holds Training Results for the Network
impl NetworkStats {
    pub fn new() -> Self {
        NetworkStats { 
            tested: 0, positive: 0, negative: 0
        }
    }
    pub fn update(&mut self, positive: bool) {
        self.tested += 1;
        if positive {  self.positive += 1}
        else { self.negative += 1 }
    }
}

impl fmt::Display for NetworkStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let div = if self.tested == 0 { 1 } else { self.tested };
        write!(f,
"Network Stats
-------------- 
tested: {}, positive: {}, negative: {}, correct: {}%",
        self.tested, self.positive, self.negative, self.positive * 100 / div)
    }
}

/// Simple normalization function
pub fn normalize(v: f64, min: f64, max: f64) -> f64 { 
    if (max - min) == 0.0 {  (v - min)  } else { (v - min) / (max - min) }
}

struct NoResultsProvided(&str);
impl Stats {
    type Error = NoResultsProvided;
    /// Returns highest scoring in the league for at home and away
    /// (all time highest scoring at home, all time highes scoring away)
    pub fn all_time_highest_score_in_league(matches: &Vec<Match>) -> [u8; 2] {
        let mut score: [u8; 2] = [0, 0];
        for m in matches {
            match m.result {
                Some(result) => {
                    if result.0 > score[0] { score[0] = result.0; }
                    else if result.1 > score[1] { score[1] = result.1;  }
                },
                None => { panic!("The results provided need Some(Resul)") }
            }
        }
        score
    }
    /// Returns the number of game days in a Vec<Matches>
    pub fn game_days(matches: &Vec<Match>) -> usize { matches.iter().fold(0, |i, _m| i + 1 ) }

    /// Creates ```Stats``` for a ```ClubName``` calculated from a Vec<Match>
    pub fn gen_stats(club: &ClubName, matches: &Vec<Match>) -> Stats {
        let mut home_scores = vec![];
        let mut away_scores = vec![];
        for m in matches {
            match m.result {
                Some(result)=> {
                    if &m.home == club {
                        home_scores.push(result.0);
                    } else if &m.away == club {
                        away_scores.push(result.1);
                    }
                },
                None => { panic!("The results provided need Some(Resul)") }
            }
        }
        Stats { home_scores, away_scores, games_played: [0, 0] }
    }
    /// Returns the alltime highest scoring of a club home or away
    /// (highest scoring for club at hone, highest scoring for club away)
    pub fn highest_alltime_scores_by_club(stats: &Stats) -> (u8, u8) {
        let mut home: u8 = 0;
        let mut away: u8 = 0;
        for score in &stats.home_scores {
            if score > &home { home = *score; }
        }
        for score in &stats.away_scores {
            if score > &away { away = *score; }
        }
        (home, away)
    }
    /// Returns the highest scoring for the home team at home and the away team away to date
    /// (highest scoring for the home team, highest scoring for the away team)
    /// Only considers matches already played (match_count);
    pub fn highest_scores_to_date(h_club: &Stats, a_club: &Stats) -> [u8; 2] {
        let mut highest_scores: [u8; 2] = [0, 0];
        if h_club.games_played[0] < h_club.home_scores.len() as u8 {
            if let Some(hh) = h_club.home_scores[..h_club.games_played[0] as usize].iter().max() {  
                highest_scores[0] = *hh;
            } else {
                highest_scores[0] =  0;
            }
        } else {
            if let Some(hh) = h_club.home_scores.iter().max(){  
                highest_scores[0] = *hh;
            } else {
                highest_scores[0] =  0;
            }
        }
         if a_club.games_played[0] < a_club.home_scores.len() as u8 {
            if let Some(ha) = a_club.away_scores[..a_club.games_played[1] as usize].iter().max() {
                highest_scores[1] = *ha;
            } else {
                highest_scores[1] =  0;
            }
         } else {
             if let Some(ha) = h_club.home_scores.iter().max() {
                highest_scores[1] = *ha;
            } else {
                highest_scores[1] =  0;
            }
         }
        
        highest_scores
    }
    /// Sums and returns the scoring for home or away matches for given 
    pub fn total_scoring_by_club_to_date(stats: &Stats, ha: Scoring) -> u8 {
        let total: u8 = if ha == Scoring::Home {
            // TODO STATS played to usize
            if stats.games_played[0] < stats.home_scores.len() as u8 {
                stats.home_scores[..stats.games_played[0] as usize].iter().sum()
            } else {
                stats.home_scores.iter().sum()
            }
        } else { 
            if stats.games_played[1] < stats.away_scores.len() as u8 {
                stats.away_scores[..stats.games_played[1] as usize].iter().sum()
            } else {
                stats.away_scores.iter().sum()
            }
        };
        total
    }
}