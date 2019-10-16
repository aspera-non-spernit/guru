use chrono::{ DateTime, FixedOffset };
use serde::{ de, Deserialize, Deserializer, Serialize, Serializer };
use std::{ collections::{ HashMap } };

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Club {
    pub name: String,
}
#[derive(Clone, Debug)]
pub struct Clubs {
    pub data: HashMap<Club, u32>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Match {
    #[serde(deserialize_with = "self::deserialize_from_str")]
    #[serde(serialize_with = "self::serialize_to_str")]
    pub date: chrono::DateTime<FixedOffset>,
    pub home: String,
    pub away: String,
    pub result: Option<[u8; 2]>,
}
#[derive(Debug)]
pub struct DataEntry {
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>
}
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
        home: String,
        away: String,
        result: Option<[u8; 2]>,
    ) -> Self {
        Match {
            date,
            home,
            away,
            result,
        }
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
