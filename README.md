# America Election Prediction in 2012

MG-GY 8401 assignment

Background In 2012 presidential election, statistician Nate Silver (of the website FiveThirtyEight) correctly predicted the outcome in every state.

The general structure of FiveThirtyEight's algorithm is:

Calculate the average error of each pollster's predictions for previous elections. This is known as the pollster's rank. A smaller rank indicates a more accurate pollster.
Transform each rank into a weight (for use in a weighted average). A larger weight indicates a more accurate pollster. FiveThirtyEight considers a number of factors when computing a weight, including rank, sample size, and when a poll was conducted. For this assignment, we simply set weight to equal the inverse square of rank (weight = rank**(-2)).
In each state, perform a weighted average of predictions made by pollsters. This predicts the winner in that state.
Calculate the outcome of the Electoral College, using the per-state predictions. The candidate with the most electoral votes wins the election.
