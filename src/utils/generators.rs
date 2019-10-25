use crate::{
    models::{Clubs, Match},
    utils::normalize,
    Features, Guru, Stats,
};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct DefaultInputGenerator<'a> {
    pub values: (Vec<Match>, &'a Clubs, HashMap<String, Stats>),
}
/**
Implementing ```Generator``` allows to pass a custom set of input features to guru
and the network.
It is used by DataEntry::from to return a set of training matches
Example:
```rust

#[derive(Clone, Debug)]
struct MyInputGen<'a> {
    values: (Vec<Match>, &'a Clubs, HashMap<String, Stats>),
}

impl<'a> MyInputGen<'a> {

}
let mut my_input_generator = MyInputGen {
    values: (training_matches.clone(), &clubs, stats.clone()),
};

```
**/
pub trait Generator {
    fn generate(&mut self, m: &Match) -> Vec<f64>;
}

impl<'a> DefaultInputGenerator<'a> {
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

impl Generator for DefaultInputGenerator<'_> {

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
        // commented: probably unneccessary
        //inputs.extend_from_slice(&Guru::club_features(m, self.values.1));

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
            Future matches (those without a result have values > 1.0)
        **/
        inputs.push(Guru::game_day(&m.date, &self.values.0));

        /***
        Adding 1 feature: League
        Takes the league field of a match and converts the String into an integer then f64
        Rationale: Allows to add a non-judgmental feature that represents the overall strength
        of the league, which may be relevant in inter league matches (ie Open Cup) or in a Pro-Rel System.
        It also allows some separation between phases of a season, if the field league is used that way.
        Example:
            The data set consists of matches from the the German Bundesliga, UK Premier Leaguge, and NISA.
            The matches of NISA show Team A to be very strong, however compared to other leagues in the
            Data set, Team A may be less successful.
            The String radix allows to add that feature without ranking the league by personal opinion.
            A first Division league in Tibet may be weaker than a a 4th Division NPSL league.
        If matches in a regular season are marked differently than for instance play-offs, friendlies
        or off-season matches, it may allow the network to pick up patterns in the roster or changing the strategy
        of the teams throughout those phases.
        Example:
            Team A tries out new formations and a more offensive play in a pre-season or a friendly, than
            in the play-offs. If there's a pattern. The network will pick that up and may be able
            to produce better predictions knowing that a result of a friendly is less reliable than a play-off result.
        **/
        // does count 5 leagues, when 6 in data set, but println! 6 leagues.
        let mut leagues: HashSet<i64> = HashSet::new();
        for m in &self.values.0 {
            leagues.insert(i64::from_str_radix(&m.league, 36).unwrap());
        }

        let hl: f64 = (*leagues.iter().max().unwrap()) as f64;
        inputs.push(normalize(
            i64::from_str_radix(&m.league, 36).unwrap() as f64,
            0f64,
            hl,
        ));

        /***
        Adding 3x2 features. The values for Home add up to 1.0 and the values for away
        add up to 1.0.
        Home and Away values are not related to each other.
        TODO:
        Adding 2 features: Home and Away WINS to date (no relation to each other)
        Sums up for the home and away them the previous matches won at home or away
        Example:
            The home team played 4 games at home, and won 3 of these.
            The away team played 4 games away, won 1 of them
            Home: 0.75 (3 of 4 matches won at home)
            Away: 0.25 (1 of 4 matches won away)

        **/
        let wins = Stats::wdl_to_date(&self.values.0, &m, std::cmp::Ordering::Greater);
        inputs.push(normalize(
            wins[0] as f64,
            0f64,
            wins[0] as f64 + wins[1] as f64,
        ));
        inputs.push(normalize(
            wins[1] as f64,
            0f64,
            wins[0] as f64 + wins[1] as f64,
        ));
        /***
        TODO:
        Adding 2 features: Home and Away DRAWS to date (no relation to each other)
        Sums up for the home and away them the previous draws at home or away
        Example:
            The home team played 4 games at home, and 1 was a draw.
            The away team played 4 games away, 2 of them were a draw
            Home: 0.25 (1 of 4 matches at home a draw)
            Away: 0.5 (1 of 4 matches away a draw)
        **/
        let draws = Stats::wdl_to_date(&self.values.0, &m, std::cmp::Ordering::Equal);
        inputs.push(normalize(
            draws[0] as f64,
            0f64,
            draws[0] as f64 + draws[1] as f64,
        ));
        inputs.push(normalize(
            draws[1] as f64,
            0f64,
            draws[0] as f64 + draws[1] as f64,
        ));
        /***
        TODO:
        Adding 2 features: Home and Away LOSSSES to date (no relation to each other)
        Sums up for the home and away them the previous matches lost at home or away
        Example:
            The home team played 4 games at home, and lost none.
            The away team played 4 games away, won 1 of them
            Home: 0.0 (0 of 4 matches lost at home)
            Away: 0.25 (1 of 4 matches lost at away)
        **/
        let losses = Stats::wdl_to_date(&self.values.0, &m, std::cmp::Ordering::Less);
        inputs.push(normalize(
            losses[0] as f64,
            0f64,
            losses[0] as f64 + losses[1] as f64,
        ));
        inputs.push(normalize(
            losses[1] as f64,
            0f64,
            losses[0] as f64 + losses[1] as f64,
        ));
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
        inputs.push(normalize(hts[0].into(), 0f64, (hts[0] + ats[1]).into()));
        inputs.push(normalize(ats[1].into(), 0f64, (hts[0] + ats[1]).into()));

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
        inputs.push(normalize(h_rel, 0f64, h_rel + a_rel));
        inputs.push(normalize(a_rel, 0f64, h_rel + a_rel));

        let hs = [
            Stats::highest_scoring_by_club_to_date(&self.values.2.get(&m.home).unwrap())[0],
            Stats::highest_scoring_by_club_to_date(&self.values.2.get(&m.away).unwrap())[1],
        ];
        /*** Adding 2 features: Relative Highest Scoring between clubs to date
        Highest scoring of the Home Team is 7
        Highest scoring of the Away Team is 4
        Values: Team A: 0,6364 Team B 0,3636
        **/

        inputs.push(normalize(hs[0].into(), 0f64, (hs[0] + hs[1]).into()));
        inputs.push(normalize(hs[1].into(), 0f64, (hs[0] + hs[1]).into()));
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
        inputs.push(normalize(hs[0].into(), 0f64, hsil[0].into()));
        inputs.push(normalize(hs[1].into(), 0f64, hsil[1].into()));

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
        let hist = self
            .values
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
        inputs.push(normalize(hist[0].into(), 0f64, (hist[0] + hist[1]).into()));
        inputs.push(normalize(hist[1].into(), 0f64, (hist[0] + hist[1]).into()));

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
