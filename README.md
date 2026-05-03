# Sentiment Analysis as a Tool for Differentiating Soccer and Football Discussions on Reddit
### EPSY 5643 — University of Connecticut, Spring 2024

## Overview
This project investigates whether sentiment analysis can reliably distinguish Reddit discussions about **soccer** from those about **American football (NFL)**. Using 11,275 posts from sport-specific subreddits, the analysis examines how fan sentiment varies by game phase (pre-game, during, post-game), match scenario (favored vs. underdog, close vs. one-sided), and win/loss outcome — then trains a classifier to predict sport from those signals.

## Dataset
- **Source**: Reddit posts scraped from Soccer and NFL subreddits
- **Full dataset**: 92,235 entries; **working set** (after filtering neutral posts): 11,275
  - Soccer: 5,781 posts | Football: 5,494 posts
- **Features**: post text, upvotes, sport label, game phase (1=pre, 2=during, 3=post), match category (favored/underdog × close/one-sided × winning/losing perspective)

## Methods
- **Text preprocessing**: emoji-to-word conversion, URL replacement, punctuation normalization, lemmatization
- **Sentiment scoring**: VADER `SentimentIntensityAnalyzer` (compound score per post)
- **Feature engineering**: TF-IDF vectorization (`min_df=5`), one-hot encoded match category, upvote count, game phase
- **Classification**: Gradient Boosting Classifier with hyperparameter tuning via `GridSearchCV` (5-fold stratified CV)

## Results
Best model (`learning_rate=0.1`, `max_depth=5`, `n_estimators=150`):

| Metric | Score |
|--------|-------|
| Accuracy | 0.7494 |
| Precision | 0.7730 |
| Recall | 0.6879 |
| F1 | 0.7280 |

**Top features by Gini importance**: game phase (0.307), upvotes (0.223), underdog category (0.121), compound sentiment (0.085)

**Key finding**: Fan sentiment is similar across both sports when the favored team wins. The largest divergence appears in **underdog scenarios** — soccer fans express more positivity when losing and more negativity when winning closely compared to football fans.

## Repository Structure
```
turn_in_doc.ipynb          # Full analysis notebook
ESPY Final Presentation.pdf  # Conference presentation slides
Medium Article 3/          # Charts and screenshots from the companion Medium article
```

## Tools & Libraries
Python — `pandas`, `numpy`, `nltk` (VADER), `scikit-learn`, `emoji`, `seaborn`, `matplotlib`
