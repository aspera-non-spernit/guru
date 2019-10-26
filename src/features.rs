use chrono::{DateTime, FixedOffset};
use crate::{
    models::{Clubs, Match},
    Stats
};

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
    fn game_day(schedule: &[Match], m: &DateTime<FixedOffset>) -> f64;
    fn goal_diff(h_stats: &mut Stats) -> f64;
}