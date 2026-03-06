# MLB Win Prediction

This project builds a machine learning model to estimate the probability that the home team will win a Major League Baseball game. Using Statcast-derived pitching, batting, and fielding metrics, the model compares recent team performance through rolling statistical windows and constructs matchup-based features that quantify differences between the two teams. Several classification algorithms—including logistic regression, random forests, and gradient boosting—are evaluated to determine which approach best predicts game outcomes. The goal of the project is to explore how recent team performance and matchup dynamics influence win probability in baseball.

## Internal TODO List

**Game Preprocessing** Notebook and Associated Scripts:

- Update order of the functions in the batter and pitching scripts to reflect their order in the notebooks
- For checking the number of games at each major point in the notebook, make a function that can work wtih all other dataframes, and rather than return the dataframe, return `True` or `False` and have an assert statement
- Double check if rolling values are computing correctly, especially for Rolling Batting and Fielding metrics
- Check why only FIP is showing for bullpen
