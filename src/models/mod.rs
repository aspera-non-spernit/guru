use chrono::{ DateTime, FixedOffset };
use serde::{ de, Deserialize, Deserializer, Serialize, Serializer };
use std::{ collections::{ HashMap }, fmt };

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum ClubName { Atlanta, California, Chattanooga, Detroit, LosAngeles, Miami, Michigan, Milwaukee, Oakland, NewYork, NapaValley, Philadelphia, SanDiego, Stumptown }
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Club { pub name: ClubName }
#[derive(Clone, Debug)]
pub struct Clubs { pub data: HashMap<Club, u32> }
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Match  {
    #[serde(deserialize_with = "self::deserialize_from_str")]
    #[serde(serialize_with = "self::serialize_to_str")]
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

fn deserialize_from_str<'de, D>(deserializer: D) -> Result<DateTime<FixedOffset>, D::Error>
where D: Deserializer<'de>, {
    let date: String = Deserialize::deserialize(deserializer)?;
    date.parse::<DateTime<FixedOffset>>().map_err(de::Error::custom)
}

fn serialize_to_str<S: Serializer>(date: &DateTime<FixedOffset>, serializer: S) -> Result<S::Ok, S::Error> {
    date.to_rfc3339().serialize(serializer)
}