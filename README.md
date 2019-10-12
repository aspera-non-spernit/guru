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
  //... more matches here
]
```

#### 2) Run guru with error rate as single parameter

```bash
$ guru 0.3579
```

#### 3) Run guru with error rate as single parameter

Wait, see the results, go bet and become rich.. :)