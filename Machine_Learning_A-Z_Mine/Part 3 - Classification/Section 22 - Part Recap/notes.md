# Decision Tree | Random Forrest
Great for Classification.
### [Deep Dive](random_forrest_notes.md)


# Ensemble Learning
Taking multiple machine learning algorithms and putting them together to create one bigger algorithm. Final algorithm uses / leverages many other ML algorithms.

# Error Types
* Type 1 | False Positive: Predicted that event would happen, when in reality it did not.
* Type 2 | False Negative: Predicted that event would not happen, when in reality it did.

# Metrics
**Accuracy Rate** = Correct / Total
**Error Rate** = Wrong / Total
**Accuracy Paradox** = Accuracy / Error rate are not insightfull enough to say if model is good.
- ex: In case when most classifications are charged to on result, it could be better to always assume the same result.

#### CAP Curve: Cumulative Accuracy Profile
One one axis goes the percent of the the population that has been targeted, on the other the % of the population that has positive engagement.
Accuracy of CAP Curve is calculated by finding out the area / integrating between the curve and the random selection process (straight line / hyperplane).

* Accuracy Ratio [AR] = a<sub>r</sub> / a<sub>p</sub>
* a<sub>r</sub> = area inbetween model curve and and random selection line
* a<sub>r</sub> = area inbetween perfect model curve and and random selection line

Rule of Thumb: Check positive outcomes taking 50% the population
* X < 60% --> Rubbish
* 60% < X < 70% --> Poor
* 70% < X < 80% --> Good
* 80% < X < 90% --> Very Good
* 90% < X < 100% --> Too Good (overfitting or other fenomena [ex: didn't filter variable that looks into the future  / direct correlation])

# [Classification Model Pro - Con Cheat sheet](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/P14-Classification-Pros-Cons.pdf)

# How do I know which model to choose for my problem ?

Same as for regression models, you first need to figure out whether your problem is linear or non linear. Then:

* If your problem is linear, you should go for Logistic Regression or SVM.
* If your problem is non linear, you should go for K-NN, Naive Bayes, Decision Tree or Random Forest.

Then from a business point of view, you would rather use:

* Logistic Regression or Naive Bayes when you want to rank your predictions by their probability. For example if you want to rank your customers from the highest probability that they buy a certain product, to the lowest probability. Eventually that allows you to target your marketing campaigns. And of course for this type of business problem, you should use Logistic Regression if your problem is linear, and Naive Bayes if your problem is non linear.
* SVM when you want to predict to which segment your customers belong to. Segments can be any kind of segments, for example some market segments you identified earlier with clustering.
* Decision Tree when you want to have clear interpretation of your model results,
* Random Forest when you are just looking for high performance with less need for interpretation.
