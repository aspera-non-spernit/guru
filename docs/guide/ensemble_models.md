### Ensemble Models

Produce a set of predictions from different Models.

#### Rationale

I tested different features, error rates etc. to predict future match results.
One model failed on almost any prediction, but one. This single correct prediction
was incorrectly predicted by two other models.

**Example**

* Model 1: Miami FC 3 : 4 Oakland Roots
* Model 2: Miami FC 2 : 3 Oakland Roots
* Model 3: Miami FC 2 : 0 Oakland Roots

Actual result:

* Miami FC 3 : 2 Oakland Roots

All three models were wrong on the actual result, Model 3 however, that failed on all other
predictions, could identify the correct winner Miami FC instead of Oakland Roots.

#### Methods

* **Averaging**: Take the average of different models. Can be used in regression problems (predict result), 
or for classification problems (predict match outcome home win, draw, away win)

```rust
Model 1 [3.0,  4.0 ]
Model 2 [2.0,  3.0 ]
Model 3 [2.0,  0.0 ]
--------------------
Average [2.33, 2.33]
```

In this example for a regression problem averaging would not identify the correct winner,
but would shift from a home loss to a draw.

* **Majority Vote**: Takes the result that was voted the most. Can only be used for classification problems
(predict winner, or draw).

This example shows three models predicting either a home win [1, 0, 0], a draw [0, 1, 0]  or an away win [0, 0, 1]

```rust
Model 1       [0, 0, 1]
Model 2       [0, 0, 1]
Model 3       [1, 0, 0]
-----------------------
Majority Vote [1, 0, 2]
```

The Majority Vote would have selected Oakland Roots as Winner, since two models predicted Oakland Roots would win,
only one model identified Miami FC as Winner.

* **Weighted Average**: Different weights are applied to the models. Can be used for both regression and
classification problems. The weights could be derived from the network error or previous performance.

|Model|Weighted Result|Weight|Predicted Result|
|:----|:--------------|:-----|:---------------|
|Model 1|         [0.9, 1.2]|0.3|[3.0, 4.0]| 
|Model 2|         [0.8, 1.2]|0.4|[2.0, 3.0]|
|Model 3|         [0.6, 0.0]|0.3|[2.0, 0.0]|
--------------------------------------------
Weighted Average  [2.3, 2.4]
```

The Weighted Avergage would shift towards a draw, with slighly higher chance for Oakland Root to win.

* **Bagging**:

* **Boosting**:

* **Stacking**:


#### Note

Note that the used models aren't really trained well and the average weights chosen randomly.

#### Summary

None of the methods of for examples would have helped to identify Miami FC as the winner of the match, 
but two methods would have shifted towards a less wrong result. 

### References

* [Ensemble Forecasting / Ensemble Models](https://en.wikipedia.org/wiki/Ensemble_forecasting)
* [How to build Ensemble Models in machine learning? (with code in R)](https://www.analyticsvidhya.com/blog/2017/02/introduction-to-ensembling-along-with-implementation-in-r/)
