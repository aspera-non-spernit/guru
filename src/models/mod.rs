use chrono::{ DateTime, FixedOffset };
use std::{ collections::{ HashMap }, fmt };


#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ClubName
{ Atlanta, California, Chattanooga, Detroit, LosAngeles, Miami, Michigan, Milwaukee, Oakland, NewYork, NapaValley, Philadelphia, SanDiego, Stumptown }
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Club { pub name: ClubName }
#[derive(Clone, Debug)]
pub struct Clubs { pub data: HashMap<Club, u32> }
#[derive(Clone, Copy, Debug)]
pub struct Match  {
    pub date: chrono::DateTime<FixedOffset>,
    pub home: ClubName,
    pub away: ClubName,
    pub result: Option<(u8, u8)>
}
#[derive(Debug, Eq, Hash, PartialEq)]
pub enum Scoring { Home, Away }

impl Club { pub fn new(name: ClubName) -> Self { Club { name } } }

impl fmt::Display for ClubName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Clubs {
    /// Returns the index 
    pub fn get_index(self, club_name: ClubName) -> u32 {
        let mut i = 0;
        for c in self.data { 
            if c.0.name == club_name {
                i = c.1;
            } else {  }  // Error should not happen at the moment. Club not in list
        }
        i
    }
}

impl Match {
    pub fn new(date: DateTime<FixedOffset>, home: ClubName, away: ClubName, result: Option<(u8, u8)>) -> Self {
        Match { date, home, away, result }
    }
}
