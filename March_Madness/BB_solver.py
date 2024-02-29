import functools
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import scipy.special as sc
import numpy as np
import math


def bb_ll(x0, args):
    """
    This is the beta-binomial function used to calculate each team's alpha and beta values for a given metric
    :param x0: [alphas + betas + home_alpha + neutral_alpha + home_beta + neutral_beta]
    :param args: [winners, losers, loc, metric_n, metric_d, recency_weights, ratings_weights, teams_n]
    :return: loss
    """

    winners = args[0]
    losers = args[1]
    location = args[2]
    k = args[3]
    n = args[4]
    recency_adj = args[5]
    strength_adj = args[6]
    n_teams = args[7]

    given_alphas = x0[0:n_teams]
    given_betas = x0[n_teams:2 * n_teams]
    home_alpha_adj = x0[-4]
    neutral_alpha_adj = x0[-3]
    home_beta_adj = x0[-2]
    neutral_beta_adj = x0[-1]

    alpha_loc_adj = [home_alpha_adj if location[i] == 1
                     else (neutral_alpha_adj if location[i] == 0 else 0) for i in range(len(location))]

    beta_loc_adj = [home_beta_adj if location[i] == -1
                    else (neutral_beta_adj if location[i] == 0 else 0) for i in range(len(location))]

    alpha = np.fromiter([given_alphas[winners[i]] + alpha_loc_adj[i] for i in range(len(winners))], dtype=float)
    beta = np.fromiter([given_betas[losers[i]] + beta_loc_adj[i] for i in range(len(losers))], dtype=float)

    numerator = sc.gammaln(alpha + k) + sc.gammaln(beta + n - k) - sc.gammaln(alpha + beta + n)
    denominator = sc.gammaln(alpha) + sc.gammaln(beta) - sc.gammaln(alpha + beta)
    scaler = np.log(np.fromiter([float(math.comb(n[i], k[i])) for i in range(len(k))], dtype=float))
    combination = -np.sum((scaler + numerator - denominator) * recency_adj * strength_adj)

    return combination


def pre_obj_fn(x0, args):
    """
    This gets the initial team rankings based on a logistic regression model
    predicting the binned point spread outcomes for each game.
    :param x0: [1.0 for _ in range(len(teams_list)+1)]
    :param args: [winners, losers, home, binned_outcome]
    :return: loss
    """

    teams_ratings_list = x0[0:-1]
    home_adv = x0[-1]
    winners = args[0]
    losers = args[1]
    home = args[2]
    outcome = args[3]

    winner_rating = [teams_ratings_list[winners[_]] + home_adv * home[_] for _ in range(len(winners))]
    loser_rating = [teams_ratings_list[loser] for loser in losers]
    teams_dict = {'winner': winner_rating, 'loser': loser_rating}
    df = pd.DataFrame(teams_dict)

    clf = LogisticRegression(random_state=0, max_iter=10000, penalty=None, multi_class='multinomial').fit(df, outcome)

    y_probs = clf.predict_proba(df)
    output = metrics.log_loss(outcome, y_probs)

    return output
