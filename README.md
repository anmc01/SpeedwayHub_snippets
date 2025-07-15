# SpeedwayHub snippets
Snippets from the SpeedwayHub project, made together with users \@kwakie13 and \@jeku001.

The project aims to develop a predictive framework for results of individual riders in the Polish speedway PGE Ekstraliga.

Key features of the projects are:
- developing an XGBoost classification model, using a thorough analysis on hyperparameter tuning
- proposing a new metric aimed to indicate the riders' predicted results -- Expected Points
- developing a new classification evaluation metric -- Accuracy with bonuses -- reflecting the speedway specifics (bonus points)
- creating speedway Elo Ratings, which are used in other sports, but never before present in speedway
- performing an extensive feature engineering, aimed to extract the most important features in the dataset or alter the weak ones

The model achieved an accuracy of over 50\% in a difficult to predict, four-class classification problem, which is a great success and a benchmark for further work.
