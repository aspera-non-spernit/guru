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
