# Intro
People who bought also bought ... That is what Association Rule Learning will help us figure out!

# Apriori

### Terms
  * **Support(M)**  = % percent of population that has characteristic M
      * M / T
  * **Confidence(M—>N)** = Percent of Population with characteristic M, that has characteristic N
      * M & N / M
  * **Lift(M —> N)** = Confidence(M —> N) / Support(M)
      * (M & N) * T / M^2
      * Lift is the improvement in the prediction

### Algorithm
    1. Set Minimum support and confidence
    2. Take all the subsets in transactions having higher support than min. support
    3. Take all the rules of these subsets that having higher confidence that the min. confidence
    4. Sort rules by dec. lift

# Eclat
Eclat model included is not really useful. Can not remove null values


### To run
1. Go to section 29 directory first:
2. run command:
`python data_preprocessing_template.py `
3. then run:
`python lib/runner.py eclat --min_support 0.02 --input_path Market_Basket_Optimisation.txt --output_path output.txt`
4. Results will be found in output.txt file
