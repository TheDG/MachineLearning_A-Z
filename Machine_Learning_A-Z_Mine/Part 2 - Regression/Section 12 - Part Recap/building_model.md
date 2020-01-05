# Building a Model(Step by Step)
- trash in, trash out
- building models --> selecting variables

### All In
    1. Through in all you variables —> usually not recommended
    2. Only if you have prior knowledge, or you have to
    3. Preparing for backward elimination

### Backward Elimination
    1. Select significance level
    2. Fit model with all possible predictors
    3. Consider the predictor with highest p-value. If its bigger than the significance level go to step 4, otherwise go to end
    4. Remove variable
    5. Fit model without variable
    6. Go back to step 3
    7. End

### Forward Selection
    1. Select significance level
    2. Fit all simple regressions models. Select the one with the lowest p value
    3. Keep this (these) variable and fit all possible models with one extra predictor
    4. Consider the predictor with the lowest p-value. IF p < significance level, got to step 3
    5. Keep the previous model (the one prev. to adding unsignificant predictor)
    6. End

### Bidirectional Elimination
    1. Select a SL to enter and to stay in the model
    2. Perform step #3-4 of forward selection (add a variable) | SL = enter
    3. Perform step #3 of backward elimination as many times as possible | SL = stay
    4. If new predictors can enter model, got to step 2
    5. End. Your model is ready

### All Possible Models
    1. Select a criterion of goodness (ex: Akaike criterion)
    2. Construct all possible models (2^n -1 models)
    3. Select the one with best criterion
