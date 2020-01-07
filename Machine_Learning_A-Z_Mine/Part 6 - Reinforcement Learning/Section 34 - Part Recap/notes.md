# Intro
Reinforcement Learning is a branch of Machine Learning, also called Online Learning. It is used to solve interacting problems where the data observed up to time t is considered to decide which action to take at time t + 1. It is also used for Artificial Intelligence when training machines to perform tasks such as walking. Desired outcomes provide the AI with reward, undesired with punishment. Machines learn through trial and error.

# Upper Confidence Bound
**N<sub>i</sub>(n)**:  the # of times arm i was selected up to this round

**R<sub>i</sub>(n)**:  the sum of rewards of arm i up to round n

**r<sub>i</sub>(n)**:  the average reward of arm i up to round n
  * r<sub>i</sub>(n) = R<sub>i</sub>(n) / N<sub>i</sub>(n)[]

**c<sub>i</sub>(n)**: Confidence interval for avg. reward of each arm at round n
 * c<sub>i</sub>(n) = [r<sub>i</sub>(n) - delta<sub>i</sub>(n), r<sub>i</sub>(n) + delta<sub>i</sub>(n) ]
 * delta<sub>i</sub>(n) =  ( 3log(n) / 2N<sub>i</sub>(n) )<sup>.5</sup>

### Algorithm
1. At each round n we calculate N<sub>i</sub>(n) and R<sub>i</sub>(n) for every arm i (from the set of all arms)
2. calculate r<sub>i</sub>(n) and c<sub>i</sub>(n) for every arm i
3. Select the arm i with the max. UCB r<sub>i</sub>(n) + delta<sub>i</sub>(n)

# Thompson Sampling
### Intuition
- Idea is that we we know that arm i has a reward y, given the following distribution: p(y|theta<sub>i</sub>)
- theta<sub>i</sub>, the probability of success, is unknown but we assume it distributes in a uniform manner --> prior distribution
- We approximate theta<sub>i</sub> by using Bayesian Interference --> posterior distribution
  - p(theta<sub>i</sub>|y) = (p(y|theta<sub>i</sub>) x p(theta<sub>i</sub>)) / p(y)
  - We get that p(theta<sub>i</sub>|y) depends on the previous amount of success and failures --> Beta function
- We solve for theta<sub>i</sub> --> we get a distribution function that depends on the posterior distribution
- At each round we take a random sample from each distribution and chose the one with the highest theta<sub>i</sub> (hight possibility of success)

**N<sub>i</sub><sup>s</sup>(n)**:  the # of times arm i got a reward up to round n

**N<sub>i</sub><sup>f</sup>(n)**:  the # of times arm i was a failure up to round n

### Algorithm
1. At each round n, we calculate N<sub>i</sub><sup>s</sup>(n) and N<sub>i</sub><sup>f</sup>(n) for every arm i (from the set of all arms)
2. For each arm i, we take a random draw from the following distribution (beta)
  * theta<sub>i</sub>(n) = B(N<sub>i</sub><sup>s</sup>(n) + 1, N<sub>i</sub><sup>f</sup>(n) +1)
3. Select the arm i with the max. theta<sub>i</sub>(n)

# UCB VS Thompson Sampling Pro - Cons | Differences
### UCB
- Deterministic
- Requires update every round

### Thompson Sampling
- Probabilistic
- Can accommodate delayed feedback
- **Has better empirical evidence**
