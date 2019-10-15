# GURU

Predict future sports matches.

## How to

#### 1) Provide a data.json that consists of an array matches

**Note:** Currently only works if data.json is located in the same folder as guru.

```json
[
  {
    "date": "2019-08-31T19:30:00-04:00",
    "home": "Detroit",
    "away": "Philadelphia",
    "result": [
      1,
      0
    ]
  },
  .. more matches here with Some(result)
  .. and matches with None aka null
  {
    "date": "2019-10-26T22:04:00-04:00",
    "home": "LosAngeles",
    "away": "California",
    "result": null
  },
]
```

Matches in the file that have a result provided are used for the training
of the network. 

Matches with no result (must be "null") will be predicted. 

#### 2) Run guru with error rate as single parameter

```bash
$ guru 0.3579
```

#### 3) Run guru with error rate as single parameter

Wait, see the results, go bet and become rich.. :)

# Benchmarks

I am testing the Network in three ways:

1. Testing against seen Training Data
2. Testing against unseen Test Data
3. Testing against unseen future Football Matches with validation after the game day.

Each of the tests above can be interpreted in two ways:

- Can the network predict the exact result of a match.
- Does the network identify the outcome of the match correctly. Home-Win, Draw, Away-Win

#### 1) Results

The results of Tests against seen Training data can be cranked up to a 100% success rate for both the
result and the outcome of the match,
by lowering the error rate.
This leads to overfitting, it does lead to slightly better results of the Tests against unseen Test Data,
not worth the disadvantages of overfitting.

The results of the Tests agains unseen Test data typically range between 16-34% for the prediction of the correct
result and 50-78% for the prediction of the correct winner or the draw for a match.

#### 2) Concrete Example:

The Data consists of all matches of the NISA Showcase Season 2019, all matches of the NPSL regular season, if 
the clubs were playing against a club that's playing in NISA or the NPSL Members Cup, and all matches of the 
NPSL Members Cup 2019.

On Oct 8-11, the network predicted the matches for the 10th week of NISA and NPSL Members Cup season.
The Match Detroit : Michigan was on Oct 8, the result was included into the training set for four remaining
matches in the week on Oct 12

| Match | Predicted Result | Actual Result |
|-------------------|------------------|---------------|
| Detroit : Michgan | 5 : 0	| 2 : 0 |
| Stumptown : Chattanooga | 0 : 1  | 2 : 2 |
| Milwaukee : Michigan | 2 : 0 | 2 : 1 |
| New York : Detroit |	1 : 1 | 0 : 0 |
| Atlanta : Miami |	0 : 2	| 1 : 2 |

##### Interpretation

Out of 5 matches the Network wasn't able to predict the correct result once.
However it was able to predict the correct winner in 4 of 5 matches. Only the match Stumptown vs. Chattanooga
was a predicted win for Chattanooga, but ended in a 2:2 draw.

The prediction of the outcome of a future match on real world data is in the range of what the test results on
unseen data suggested. In this particular example 80%. Due to the small number of matches a second false prediction,
the success reate would drop to 60% (3 out of 5 matches).

It is expected that on a game day with much more matches, the success rate would remain in that range.