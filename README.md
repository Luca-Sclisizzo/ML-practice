This repository contains all my little trainings and tutotal on ML prediction models.
1. RF.regression.housing.py: is a script built on sklearn library. The model is fitted to predict the value of the houses in California with 8 features.
   Then I decided to select the most important features for the prediction (with PVI). Whith these two I displayed a surface plot.
2. RF.regression.housing.corrected: is the corrected versiono of 1st script. I realized that the inflated R² in my model was due to a conceptual mistake: I was evaluating the
   model's performance using the same dataset that I trained it on. This can lead to overfitting, where the model learns patterns specific to the training data, resulting in an
   artificially high R².
   To correct this, I split the feature_df dataset into separate training and testing sets. By training the model on one portion of the data and evaluating it on the other, I
   ensured that the R² value reflects the model's true ability to generalize to new, unseen data, providing a more accurate and realistic measure of its predictive performance.
   Indeed, I added a new section to train the model in the correct way; then I analyzed also the R^2 of the 3 models.
