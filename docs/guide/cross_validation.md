### Cross Validation

Use multiple sub sets of a data set D to train and test the network.

#### Rationale

If the available data set is small, spliiting the data set into a trainning and a test set
may cause the network to underfit due to lack of data.

#### Method

A relative small data set D is split into sub sets of size D/k or D/k+1 , if possible and randomly picked
items of the data set D.
One set is used to test the network all other sets are used for training.
The test set changes with each iteration through the sub sets.

There are k rounds of training and testing.

**Example**

```rust
k = 3
data_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
sets = [
    [9, 4, 6, 2],
    [1, 10, 8],
    [3, 7, 5]
]

// three rounds of training and testing

// round 1
training: sets[0] + sets[1]
testing: sets[2]

// round 2
training: sets[1] + sets[2]
testing: sets[0]

// round 2
training: sets[2] + sets[0]
testing: sets[1]
```