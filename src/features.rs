use chrono::{DateTime, FixedOffset};
use crate::{
    Stats,
    models::{Match, Scoring},
    utils::{normalize}
};
use std::{
    collections::{HashSet},
};

/**
    Calculates the distance from the date of the match to the earliest and newest match date in data set.
    The match date is converted into a Unix Timestamp. The earliest and lates match dates as timestamp
    act as min and max for the normalization.
    
    **Rationale**:

    * More recent matches a valued higher than earlier matches.

    **Example**:

    * The earliest match in the data set is on 2019-05-12 Unix Timestamp 1557619200
    * Date of the current match is 2019-10-02 Unix Timestamp 1569974400
    * The last match in the data set is on 2019-11-02 Unix Timestamp 1572652800
    * The value for the game day factor is 0.8218 (a relative recent match)
    * Future matches (those without a result have values > 1.0)
**/
pub struct GameDayFeature { pub data: f64 }
impl From<(&[Match], &DateTime<FixedOffset>)> for GameDayFeature {
    fn from(from: (&[Match], &DateTime<FixedOffset>) ) -> Self {
        let mm: HashSet<i64> = from.0.iter()
            .map(|s| s.date.timestamp())
            .collect();
        let data = normalize(
                from.1.timestamp() as f64,
                *mm.iter().min().unwrap() as f64,
                *mm.iter().max().unwrap() as f64
        );
        GameDayFeature{ data }
    }
}
/**
    Returns goal difference a Club shot at home and away
    Note: Stats must be updated before ```from``` is invoked.

    **Example**:

    * Team A played 4 matches at home, score [3, 0, 2, 1] = 6
    * Team A played 2 matches away, scored [1, 1] = 2
    * Scoring::Home = 3.0
    * Scoring::Away = 0.33
    
    **Intepretation**:

    * Team A shot three times more goals at home than away
    * Team A shot 1/3 the goals away, it has shot a home.

    TODO:
    
    * move to Stats::goal_diff
    * Return normalized Home + Away Goal Diff
**/
pub struct GoalDiffFeature { pub data: f64 }
impl From<(&mut Stats, Scoring)> for GoalDiffFeature {

    fn from(from: (&mut Stats, Scoring)) -> Self {

        let h = from.0
            .home_scores
            .drain(0..from.0.games_played[0] as usize)
            .collect::<Vec<u8>>()
            .iter()
            .sum::<u8>();
        let a = from.0
            .away_scores
            .drain(0..from.0.games_played[1] as usize)
            .collect::<Vec<u8>>()
            .iter()
            .sum::<u8>();
        match from.1 {
            Scoring::Home => {
                if a != 0 {
                    GoalDiffFeature { data: h as f64 / a as f64 }
                } else {
                    GoalDiffFeature { data: f64::from(h) }
                }
            },
            Scoring::Away => {
                if h != 0 {
                    GoalDiffFeature { data: a as f64 / h as f64 }
                } else {
                    GoalDiffFeature { data: f64::from(a) }
                }
            }
        }
        
    }
} 
/**
    Used if the data set contains matches from various leagues and inter league matches.
    Assigns a str_radix to designate each league.

    Takes the league field of a match and converts the String into an integer then f64
    Rationale: Allows to add a non-judgmental feature that represents the overall strength
    of the league, which may be relevant in inter league matches (ie Open Cup) or in a Pro-Rel System.
    It also allows some separation between phases of a season, if the field league is used that way.
    
    **Example**:

    * The data set consists of matches from the the German Bundesliga, UK Premier Leaguge, and NISA.
    * The matches of NISA show Team A to be very strong, however compared to other leagues in the
    Data set, Team A may be less successful.

    The String radix allows to add that feature without ranking the league by personal opinion.
    A first Division league in Tibet may be weaker than a 4th Division NPSL league.
    
    If matches in a regular season are marked differently than for instance play-offs, friendlies
    or off-season matches, it may allow the network to pick up patterns in the roster or changing the strategy
    of the teams throughout those phases.
    
    **Example**:
    
    * Team A tries out new formations and a more offensive play in a pre-season or a friendly, than
    in the play-offs. If there's a pattern. The network will pick that up and may be able
    to produce better predictions knowing that a result of a friendly is less reliable than a play-off result.
**/
pub struct LeagueFeature { pub data: f64 }
impl From<(&[Match], &Match)> for LeagueFeature {
    fn from(from: (&[Match], &Match)) -> Self {
        let mut leagues: HashSet<i64> = HashSet::new();
        for m in from.0 {
            leagues.insert(i64::from_str_radix(&m.league, 36).unwrap());
        }
        let hl: f64 = (*leagues.iter().max().unwrap()) as f64;

        let data = normalize(
            i64::from_str_radix(&from.1.league, 36).unwrap() as f64,
            0f64,
            hl,
        );
        LeagueFeature { data }
    }   
}
/**
    Returns the Median Home Score for the home team at home and the away team away
    as normalized value pair.

    **Example**:

    * Home team played 5 matches at home and scored: [2, 3, 4, 1, 4], sorted [1, 2, 3, 4, 4]
    * Away team played 4 matches away and scored: [2, 0, 1, 2], sorted [0, 1, 2, 2]
    * The Median Home score for the Home team is 3
    * The median Away Score for the Away Team is: 1.5 
    * Return MedianScoreFeature [ data: [0.6667, 0.3333] ]
**/
pub struct MedianScoreFeature { pub data: [f64; 2] }
impl From<(&[Match], &Match)> for MedianScoreFeature {
    fn from(from: (&[Match], &Match)) -> Self {
        //dbg!(&from.1);
        let matches: Vec<&Match> = from.0.iter()
            .filter(|m|
                m.date < from.1.date &&
                m.home == from.1.home ||
                m.date < from.1.date &&
                m.away == from.1.away &&
                m.result.is_some()
            )
            .collect();
        let mut scores: [Vec<u8>; 2] = [ vec![], vec![] ];
        for m in matches {
            if m.home == from.1.home {
                scores[0].push(m.result.unwrap()[0]);
            } 
            else if m.away == from.1.away {
                scores[1].push(m.result.unwrap()[1]);
            }
        }
        scores[0].sort();
        scores[1].sort();
        //dbg!(&scores);

        let mut medians: [f64; 2] = [0f64, 0f64];
       
        for i in 0..=1 {
            if scores[i].len() == 1 {
                 medians[i] = scores[i][0].into();
            } else if scores[i].len() > 1 {
                if scores[i].len() % 2 == 0 { // even len
                    medians[i] = ( scores[i][scores[i].len() / 2 - 1] + scores[i][scores[i].len() / 2] ) as f64 / 2f64;
                } else { // odd len
                    medians[i] = scores[i][(scores[i].len() - 1) / 2].into();
                }
            } else { } // len 0
        }
        //dbg!(&medians);
        let data = [
            normalize(
                medians[0] as f64,
                0f64,
                medians[0] as f64 + medians[1] as f64,
            ),
            normalize(
                medians[1] as f64,
                0f64,
                medians[0] as f64 + medians[1] as f64,
            ),
        ];
        //dbg!(&data);
        MedianScoreFeature { data }
    }
}
/**
    Returns home wins, draws or losses for the home team and away wins, draws or losses
    for the away team. Values are normalized data[0] + data[1] = 1.ß
    and the away team away.

    **Note**:

    * Ordering::Greater is for Wins
    * Ordering::Less for Losses
    * Ordering::Equal for Draws
    
    **Exanple**:

    * Asking for Wins: Ordering::Greater
    * Home team won 5 match at home
    * Away team won 1 match away
    * WDLFeature { data: [0.8333, 0.1667] }
**/
pub struct WDLFeature { pub data: [f64; 2] }
impl From< (&[Match], &Match, std::cmp::Ordering)> for WDLFeature {
    fn from(from: (&[Match], &Match, std::cmp::Ordering)) -> Self {
         let h = from.0.iter()
            .filter(|m|
                m.date < from.1.date &&
                m.home == from.1.home &&
                m.result.is_some()
            )     
            .map(|m| m.result.unwrap() )
            .map(|r| r[0].cmp(&r[1]) )
            .filter(|o| o.eq(&from.2) )
            .count();
        let a = from.0.iter()
            .filter(|m|
                m.away == from.1.away && 
                m.date < from.1.date &&
                m.result.is_some()
            )     
            .map(|m| m.result.unwrap() )
            .map(|r| r[0].cmp(&r[1]) )
            .filter(|o| o.eq(&from.2) )
            .count();
        let data = [
            normalize(
                h as f64,
                0f64,
                h as f64 + a as f64,
            ),
            normalize(
                a as f64,
                0f64,
                h as f64 + a as f64,
            )
        ];
        WDLFeature { data }
    } 
}