use crate::{
    features::{GameDayFeature, LeagueFeature, MedianScoreFeature, WDLFeature},
    models::{Clubs, Match},
    utils::normalize,
    Stats,
};
use std::collections::{HashMap};

/***
Example implementation
**/
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

        // Adding 1 feature: GameDay
        inputs.push(GameDayFeature::from( (self.values.0.as_slice(), &m.date) ).data);

        // Adding 1 feature: GoalDiff
        // inputs.push(GoalDiffFeature::from( &mut self.values.2 ).data);

        // Adding 1 feature: League
        inputs.push(LeagueFeature::from( (self.values.0.as_slice(), m) ).data);
        
        // Adding 3 x 2 features: WDLFeature
        // Home and Away Wins
        inputs.extend_from_slice(&WDLFeature::from( (self.values.0.as_slice(), m, std::cmp::Ordering::Greater) ).data);
        // Home and Away Draws
        inputs.extend_from_slice(&WDLFeature::from( (self.values.0.as_slice(), m, std::cmp::Ordering::Equal) ).data);
        // Home and Away Losses
        inputs.extend_from_slice(&WDLFeature::from( (self.values.0.as_slice(), m, std::cmp::Ordering::Less) ).data);

        // Adding 1 feature: MedianScore
        inputs.extend_from_slice(&MedianScoreFeature::from ( (self.values.0.as_slice(), m) ).data );

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
