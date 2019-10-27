use chrono::{DateTime, FixedOffset};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;

/**
Represents an arbitrary Club.
guru will create a club, giving a well-formed set of matches (example/data.json)
**/
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Club {
    pub name: String,
}

/**
All clubs that were found in a set of matches.
**/
#[derive(Clone, Debug)]
pub struct Clubs {
    pub data: HashMap<Club, u32>,
}

/**
One item of a training, test or prediction set, that consists of a set of input features and output features.
**/
#[derive(Debug)]
pub struct DataEntry {
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>,
}

/**
Represents a Match between two teams.

**Note**:

* Clubs are represented by a ```String``` not by the struct ```Club```

**/
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Match {
    #[serde(deserialize_with = "self::deserialize_from_str")]
    #[serde(serialize_with = "self::serialize_to_str")]
    pub date: chrono::DateTime<FixedOffset>,
    pub league: String,
    pub home: String,
    pub away: String,
    pub result: Option<[u8; 2]>,
}

#[derive(Debug)]
pub struct Sets<'a> {
    original: Option<&'a [Match]>,
    data: Vec<Vec<Match>>
}
/**
Can be used to mark either the scoring of the home team or the away team.
**/
#[derive(Debug, Eq, Hash, PartialEq)]
pub enum Scoring {
    Home,
    Away,
}

impl Club {
    pub fn new(name: String) -> Self {
        Club { name }
    }
}

impl Clubs {
    /// Returns the index
    pub fn get_index_by_name(self, club_name: String) -> u32 {
        let mut i = 0;
        for c in self.data {
            if c.0.name == club_name {
                i = c.1;
            } else {
            } // Error should not happen at the moment. Club not in list
        }
        i
    }
}

impl Match {
    pub fn new(
        date: DateTime<FixedOffset>,
        league: String,
        home: String,
        away: String,
        result: Option<[u8; 2]>,
    ) -> Self {
        Match {
            league,
            date,
            home,
            away,
            result,
        }
    }
}

impl <'a>Sets<'a> {
    pub fn new(original: Option<&'a[Match]>, data: Vec<Vec<Match>>) -> Self {
        Sets { original, data }
    }
}

fn deserialize_from_str<'de, D>(deserializer: D) -> Result<DateTime<FixedOffset>, D::Error>
where
    D: Deserializer<'de>,
{
    let date: String = Deserialize::deserialize(deserializer)?;
    date.parse::<DateTime<FixedOffset>>()
        .map_err(de::Error::custom)
}

fn serialize_to_str<S: Serializer>(
    date: &DateTime<FixedOffset>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    date.to_rfc3339().serialize(serializer)
}
