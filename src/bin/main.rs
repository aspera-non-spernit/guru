#![forbid(unsafe_code)]
#[macro_use]
extern crate clap;
extern crate guru;

use clap::{ App };
use guru::{
    models::{Clubs, DataEntry, Match},
    neural::nn::NN,
    utils::{load_matches, load_network, save_network},
    Features, Generator, Guru, Markdown, Stats, Testing, Training,
};
use std::{collections::HashMap, str::FromStr};

fn stats(clubs: &Clubs) -> HashMap<String, Stats> {
    let mut league_stats = HashMap::new();
    for c in &clubs.data {
        league_stats.insert(c.0.name.clone(), Stats::default());
    }
    league_stats
}

#[derive(Clone, Debug)]
struct MyInputGen<'a> {
    values: (Vec<Match>, &'a Clubs, HashMap<String, Stats>),
}

impl<'a> MyInputGen<'a> {
    /***
    Updates the Stats for the home and away team
    Adds the respective goals in the result of the match to the home scores
    for the home team and ...
    Increments games_played[0] for the home team of the match and games_played[1]
    for the away team of this match.
    **/
    fn update(&mut self, m: &Match) {
        let mut h_stats = self.values.2.get_mut(&m.home).unwrap().clone();
        let mut a_stats = self.values.2.get_mut(&m.away).unwrap().clone();
        // no updating results, if prediction
        if let Some(result) = m.result {
            h_stats.home_scores.push(result[0]);
            h_stats.games_played[0] += 1;
            a_stats.away_scores.push(result[1]);
            a_stats.games_played[1] += 1;
        }
    
        self.values.2.remove(&m.home);
        self.values.2.insert(String::from(&m.home), h_stats.clone());
        self.values.2.remove(&m.away);
        self.values.2.insert(String::from(&m.away), a_stats.clone());
    }
}
impl Generator for MyInputGen<'_> {
    // 0 training_matches, 1 &clubs, 2 &stats
    fn generate(&mut self, m: &Match) -> Vec<f64> {
        let mut inputs = vec![];

        /*** Adding 14 features: Clubs
        The data set consists of c clubs (here: 14)
        In a Vec of len 14 each each slot is reserved for one club throughout the training
        Playing clubs are given a value of 1.0, non playing clubs are given a value of 0.0
        The normalized value 0.5 for each team
        Currently not used: a global AWAY FACTOR (or AWAY_DISADVANTAG, HOME_ADVANTAGE),
        where home teams are given a value of 1.0, and away teams a afraction of 1.0 (ie. 0.8).
        The normalized values for the home team 0.5556 (=1/1.8) and the away team 0.4444 (=0.8/1.8)
        A Away team 80% as strong as a Home Team.
        **/
        inputs.extend_from_slice(&Guru::club_features(m, self.values.1));

        /***
        Adding 1 feature: Game Day Factor
        Calculates the distance from the date of the match to the earliest and newest match date in data set.
        The match date is converted into a Unix Timestamp. The earliest and lates match dates as timestamp
        act as min and max for the normalization.
        The Idea: More recent matches a valued higher than earlier matches
        Example:
            The earliest match in the data set is on 2019-05-12 Unix Timestamp 1557619200
            Date of the current match is 2019-10-02 Unix Timestamp 1569974400
            The last match in the data set is on 2019-11-02 Unix Timestamp 1572652800
            The value for the game day factor is 0.8218 (a relative recent match)
        **/
        inputs.push(Guru::game_day(&m.date, &self.values.0));

        /***
        Adding 2 features : Total Scorings to Game Day
        Calculates the relative strength of home and away team based on total scorings to game date
        Example:
            The home team scored at home in three matches [2, 3, 2]. The total scoring to date is 7
            The away team sored away in 4 matches [0, 1, 1, 2]. The total scoring is 4.
            Max for the normalization is 11
            The normalized strength for
            Home: 0,6363
            Away: 0,3636
        **/
        let hts = Stats::total_scoring_by_club_to_date(&self.values.2.get(&m.home).unwrap());
        let ats = Stats::total_scoring_by_club_to_date(&self.values.2.get(&m.away).unwrap());
        inputs.push(guru::utils::normalize(
            hts[0].into(),
            0f64,
            (hts[0] + ats[1]).into(),
        ));
        inputs.push(guru::utils::normalize(
            ats[1].into(),
            0f64,
            (hts[0] + ats[1]).into(),
        ));

        /***
        Adding 2 features : Relative Home Advantage
        Calculates the relative strength based on the home and away team's home and way (dis-)advtange
        Examples:
            The Home team scored to date and at home 7 goals, 6 goals away. Home Strength 1.6667
            The Away team scored to date and at home 4 goals, and 1 goal away. Away Strength 0.25
            The relative normalized strengths between to teams: Home 0.82353 + Away 0.17647 = 1.0
            If teams played a second match home - away flipped and assuming no goals shot in first match:
            The former Home team scored to date and at home 7 goals, 6 goals away. Away Strength 0.8571
            The former Away team scored to date and home 4 goals, and 1 goal away. Home Strength 4.0
            The relative normalized strengths between to teams: former Home 0,1765 + Away 0,8235 = 1.0
        **/
        let h_rel: f64 = if f64::from(hts[1]) != 0f64 {
            f64::from(hts[0]) / f64::from(hts[1])
        } else {
            f64::from(hts[0])
        };
        let a_rel: f64 = if f64::from(ats[0]) != 0f64 {
            f64::from(ats[1]) / f64::from(ats[0])
        } else {
            f64::from(ats[1])
        };
        inputs.push(guru::utils::normalize(h_rel, 0f64, h_rel + a_rel));
        inputs.push(guru::utils::normalize(a_rel, 0f64, h_rel + a_rel));

        let hs = [
            Stats::highest_scoring_by_club_to_date(&self.values.2.get(&m.home).unwrap())[0],
            Stats::highest_scoring_by_club_to_date(&self.values.2.get(&m.away).unwrap())[1],
        ];
        /*** Adding 2 features: Relative Highest Scoring between clubs to date
        Highest scoring of the Home Team is 7
        Highest scoring of the Away Team is 4
        Values: Team A: 0,6364 Team B 0,3636
        **/

        inputs.push(guru::utils::normalize(
            hs[0].into(),
            0f64,
            (hs[0] + hs[1]).into(),
        ));
        inputs.push(guru::utils::normalize(
            hs[1].into(),
            0f64,
            (hs[0] + hs[1]).into(),
        ));
        /*** Adding 2 features: Team's Highest Score to League Performance (Highest Scoring)
        Calculates two individual values for the home and away team relative to the league's performance
        Example:
            Highest scoring for the Home team at home is to date 6 goals
            The highest scoring for a Home team at home in the league is to date 8 goals
            The relative strength of the home team to the best home performer in the league: 0.75
            Highest scoring for the Away team away is to date 2 goals
            The highest scoring for a Away team away in the league was to date 4 goals
            The relative strength of the Home team to the best Home performer in the league: 0.5
            Features are not related to each other, do not add up to 1.0
        **/
        let hsil = Stats::highest_scoring_in_league_to_date(&self.values.0, &m.date);
        inputs.push(guru::utils::normalize(hs[0].into(), 0f64, hsil[0].into()));
        inputs.push(guru::utils::normalize(hs[1].into(), 0f64, hsil[1].into()));

        //TODO:
        /*** Adding 2 features: Team's match history
        Sums up all goals for matches in which both teams played against eachother.
        Calculates the relative strength as normalized value of total goals.
        Example:
            Team A and Team B played three times against each other:
            Scorings per match: Team A [1, 3, 2] Team B [ 0, 1, 1]
            Total Scores: Team A: 6 Team B 2:
            Normalized values: Team A 0.75 Team B 0.25
        **/
        let hist = self.values
            .0
            .iter()
            .filter(|n| {
                m.home == n.home && m.away == n.away || m.home == n.away && m.away == n.home
            })
            .filter(|n| n.date < m.date)
            .filter(|n| n.result.is_some())
            .fold([0; 2], |hist, n| {
                if m.home == n.home {
                    [
                        hist[0] + n.result.unwrap()[0],
                        hist[1] + n.result.unwrap()[1],
                    ]
                } else {
                    [
                        hist[0] + n.result.unwrap()[1],
                        hist[1] + n.result.unwrap()[0],
                    ]
                }
            });
        
        // if m.home == "Detroit City FC" && m.away == "New York Cosmos B" ||
        //     m.away == "Detroit City FC" && m.home == "New York Cosmos B" {
        //         println!("{:?}", m);
        //         dbg!(&hist);
        // }
        inputs.push(guru::utils::normalize(hist[0].into(), 0f64, (hist[0] + hist[1]).into()));
        inputs.push(guru::utils::normalize(hist[1].into(), 0f64, (hist[0] + hist[1]).into()));

        // Updating Stats
        self.update(&m);

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
        inputs
    }
}

fn main() -> std::io::Result<()> {
    let yaml = load_yaml!("/home/genom/Development/guru/cli.yml");
    let opts = App::from_yaml(yaml).get_matches();
    let error = f64::from_str(opts.value_of("error").unwrap()).unwrap();
    let all_matches = if let Some(f) = opts.value_of("data") {
        load_matches(f)?
    } else {
        // example matches
        load_matches("examples/data.json")?
    };
    // must be sorted, doesn't matter to what,
    // network needs the same order otherwise the nwtwork
    // would not be able to calculate error correctly for networks loaded from file
    // TODO: avoid sorted, collect into all_matches
    let mut sorted: Vec<Match> = all_matches.iter().cloned().collect();
    sorted.sort_by(|a, b| a.date.cmp(&b.date).to_owned() );
    // Clubs is required because ```Club```(s) are taken from a set of matches (data.json) without
    // ids
    let clubs: Clubs = Clubs::from(sorted.as_slice());
    let stats = stats(&clubs);
    let guru = Guru::new(&sorted);

    let mut training_matches: Vec<Match> = sorted
        .iter()
        .filter(|&m| m.result.is_some())
        .cloned()
        .collect();
    // taking from training_matches for testing
    let test_matches: Vec<Match> = training_matches.drain(37..training_matches.len()).collect();
    // using matches in the data set that have no result (match in the future) to predict the result 
    // for those matches
    let prediction_matches: Vec<Match> = sorted
        .iter()
        .filter(|&m| m.result.is_none())
        .cloned()
        .collect();

    // required for normalization of results (output)
    let ats = Stats::all_time_highest_score_in_league(&sorted);
    // TODO: let Generator do that
    let max = if ats[0] > ats[1] { ats[0] } else { ats[1] };

    let mut my_in_gen = MyInputGen {
        values: (training_matches.clone(), &clubs, stats.clone()),
    };

    let training_set: Vec<DataEntry> = training_matches.iter()
        .map(|m| {
            DataEntry::from( (m, &clubs, max, &mut my_in_gen) )
        })
        .collect();
    let test_set: Vec<DataEntry> = test_matches.iter()
        .map(|m| {
            DataEntry::from( (m, &clubs, max, &mut my_in_gen) )
        })
        .collect();
    let prediction_set: Vec<DataEntry> = prediction_matches.iter()
        .map(|m| {
            DataEntry::from( (m, &clubs, max, &mut my_in_gen) )
        })
        .collect();

    // Creating the network
    //let _hidden_size = (training_set[0].inputs.len() as f64 * 0.66).round() as u32;
    let mut net = if opts.is_present("load-network") {
        load_network()?
    } else {
        NN::new(&[
            training_set[0].inputs.len() as u32,
            15,
            9,
            training_set[0].outputs.len() as u32,
        ])
    };

    println!("Training Prediction Network...");
    // training
    guru.train(
        &mut net,
        &training_set,
        0.3,
        0.2,
        error,
    );
    if opts.is_present("save-network") { save_network(&net)?; }

    // testing / validating
    let (test_results, predictions) = guru.test(
        &mut net,
        &training_set,
        &training_matches,
    );
    println!("Testing on (seen) Training Data");
    println!("{}", predictions);
    println!("Result {}\n", test_results[0].to_string());
    println!("Winner {}", test_results[1].to_string());
    println!("--------------------------\n\n");
    let (test_results, test_predictions) = guru.test(
        &mut net,
        &test_set,
        &test_matches,
    );
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


