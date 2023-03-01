#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pandas as pd
import sklearn as skl
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import csl


####################################
# DATA                             #
####################################
def generate_toy_data(n=100000, seed=12345, outcome_type="binary"):
    """
    Function for generating toy data. Originally written by Hadi with slight changes for reproducibility and adding in an option for a continuous response.
    The reproducibility change is to use a numpy rng and use a seed for the random number generation
    """
    rng = np.random.default_rng(seed)
    probs = [rng.uniform() for i in range(n)]
    race = rng.binomial(1, probs)
    race = [1 if rng.random() < probs[i] else 0 for i in range(n)]
    features_b = [
        rng.multivariate_normal(
            [0.1, 0.2],
            [[0.025, -0.02], [-0.02, 0.05]],
            size=None,
            check_valid="warn",
            tol=1e-8,
        )
        for i in range(n)
    ]
    features_nb = [
        rng.multivariate_normal(
            [0.2, 0.1],
            [[0.025, 0.02], [0.02, 0.05]],
            size=None,
            check_valid="warn",
            tol=1e-8,
        )
        for i in range(n)
    ]
    features_0 = [
        features_b[i][0] if race[i] == 1 else features_nb[i][0]
        for i in range(len(race))
    ]
    features_1 = [
        features_b[i][1] if race[i] == 1 else features_nb[i][1]
        for i in range(len(race))
    ]
    # For binary outcomes
    if outcome_type == "binary":
        outcome_probs = [
            min(
                1,
                max(features_0[i] + features_1[i] + features_0[i] * (race[i] == 1), 0),
            )
            for i in range(len(race))
        ]
        outcomes = rng.binomial(1, outcome_probs)
    else:
        # For continuous outcomes
        noise = rng.normal(scale=0.01, size=len(race))
        outcomes = [
            features_0[i] + features_1[i] + features_0[i] * (race[i] == 1) + noise[i]
            for i in range(len(race))
        ]

    dataset = pd.DataFrame(
        {
            "p_black": probs,
            "race": race,
            "x0": features_0,
            "x1": features_1,
            "label": outcomes,
        }
    )
    return dataset


def split_data(dataset, test_size=0.33, random_state=42):
    X = dataset[["x0", "x1"]].to_numpy()
    Y = dataset["label"].to_numpy()
    b_pred = dataset["p_black"].to_numpy()
    race = dataset["race"].to_numpy()
    (
        X_train,
        X_test,
        y_train,
        y_test,
        b_train,
        b_test,
        b_t_train,
        b_t_test,
    ) = train_test_split(X, Y, b_pred, race, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test, b_train, b_test, b_t_train, b_t_test


dataset = generate_toy_data(outcome_type="binary")
X_train, X_test, y_train, y_test, b_train, b_test, b_t_train, b_t_test = split_data(
    dataset
)


class SyntheticData:
    def __init__(self, data, target, prob_feat, true_feat):
        self.data = data
        self.target = target
        self.prob_feat = prob_feat
        self.true_feat = true_feat

    def __getitem__(self, index):
        if type(index) is int:
            data, target, prob_feat, true_feat = (
                self.data[[index]],
                self.target[[index]],
                self.prob_feat[[index]],
                self.true_feat[[index]],
            )
        else:
            data, target, prob_feat, true_feat = (
                self.data[index],
                self.target[index],
                self.prob_feat[index],
                self.true_feat[index],
            )

        # Squeeze if single data point
        if len(data.shape) == 1:
            data = data.squeeze()
            target = target.squeeze()
            prob_feat = prob_feat.squeeze()
            true_feat = true_feat.squeeze()

        return data, target, prob_feat, true_feat

    def __len__(self):
        return self.target.shape[0]


train_data = SyntheticData(
    torch.from_numpy(X_train).to(torch.float),
    torch.from_numpy(y_train).to(torch.long),
    torch.from_numpy(b_train).to(torch.float),
    torch.from_numpy(b_t_train).to(torch.float),
)

test_data = SyntheticData(
    torch.from_numpy(X_test).to(torch.float),
    torch.from_numpy(y_test).to(torch.long),
    torch.from_numpy(b_test).to(torch.float),
    torch.from_numpy(b_t_test).to(torch.float),
)

####################################
# MODEL                            #
####################################
class Logistic:
    def __init__(self, n_features):
        self.parameters = [
            torch.zeros(1, dtype=torch.float, requires_grad=True),  # intercept
            torch.zeros(
                [n_features, 1], dtype=torch.float, requires_grad=True
            ),  # betas
        ]

    def __call__(self, x):
        yhat = self.logit(torch.mm(x, self.parameters[1]) + self.parameters[0])

        return torch.cat((1 - yhat, yhat), dim=1)

    def predict(self, x):
        _, predicted = torch.max(self(x), 1)
        return predicted

    @staticmethod
    def logit(x):
        return 1 / (1 + torch.exp(-x))


####################################
# PROBLEM                          #
####################################
class fairClassification(csl.ConstrainedLearningProblem):
    def __init__(self, data, rhs=None):
        self.model = Logistic(data[0][0].shape[1])
        self.data = data
        self.obj_function = self.loss

        if rhs is not None:
            self.constraints = [
                self.DisparityEstimate(self),
                # self.CondtlCovariance(self),
            ]
            self.rhs = rhs

        super().__init__()

    def loss(self, batch_idx):
        # Evaluate objective
        x, y, prob_feat, true_feat = self.data[batch_idx]
        yhat = self.model(x)

        # the original code regularizes the betas here but for the toy example we only have two features so it seems a bit unnecessary?
        return F.cross_entropy(yhat, y) + 0 * (
            self.model.parameters[0] ** 2 + self.model.parameters[1].norm() ** 2
        )

    class DisparityEstimate:
        def __init__(self, problem):
            """
            problem: Instantiation of the class that calls this constraint class. This is needed for some reason because we are making use of the __call__ function.
            """
            self.problem = problem

        def __call__(self, batch_idx, primal):
            x, y, prob_feat, true_feat = self.problem.data[batch_idx]

            b = prob_feat
            b_bar = torch.mean(b)

            # if primal:
            # not sure whether we need the primal flag
            if primal:
                yhat = self.problem.model(x)
                d_l = torch.sum(
                    ((yhat[:, 1] - torch.mean(yhat[:, 1])) * (b - b_bar))
                ) / torch.sum((b - b_bar) ** 2)
            else:
                yhat = self.problem.model(x)
                d_l = torch.sum(
                    ((yhat[:, 1] - torch.mean(yhat[:, 1])) * (b - b_bar))
                ) / torch.sum((b - b_bar) ** 2)
            return d_l

    class CondtlCovariance:
        def __init__(self, problem):
            """
            problem: Instantiation of the class that calls this constraint class. This is needed for some reason because we are making use of the __call__ function.
            """
            self.problem = problem

        def __call__(self, batch_idx, primal):
            """
            prob_feat: array of the probabilistic estimates of the protected feature. Should be of shape (x.shape[0], 1)
            true_feat: array of the observed values of the protected feature. Should be of shape (x.shape[0], 1)
            """
            x, y, prob_feat, true_feat = self.problem.data[batch_idx]

            B = true_feat
            B_unique = torch.unique(B)

            b = prob_feat

            yhat = self.problem.model(x)

            # not sure whether we need to use the primal flag since our constraints are differentiable
            # calculate the expected covariance of yhat and b conditional on B.
            if primal:
                expect_cov = 0.0
                for val in B_unique:
                    expect_cov += self.calc_condtl_exp(yhat[:, 1], b, B, val)
            else:
                expect_cov = 0.0
                for val in B_unique:
                    expect_cov += self.calc_condtl_exp(yhat[:, 1], b, B, val)
            # our constraint should be of form g(x) <= 0 so if we want the covariance to be positive then we need to return -g(x) but if we want covariance to be negative we want g(x).
            return -expect_cov

        def calc_condtl_exp(self, yhat, prob_feat, true_feat, true_feat_value):
            # overly complex way to compute mean since pytorch doesn't let you take mean of boolean values??
            prob_emp = (true_feat == true_feat_value).sum().div(true_feat.shape[0])
            Y_hat = yhat[true_feat == true_feat_value]
            b = prob_feat[true_feat == true_feat_value]
            b_bar = torch.mean(b)
            return (
                prob_emp
                * (1 / torch.sum(true_feat == true_feat_value))
                * torch.sum(Y_hat * (b - b_bar))
            )


# The first argument for rhs is the rhs value for disparity and the second position is for the conditional covariance.
# if we pass rhs=None then it is equivalent to solving the unconstrained problem
problems = {
    "unconstrained": fairClassification(train_data),
    "constrained": fairClassification(train_data, rhs=[0.05]),
}


#%% ################################
# TRAINING                         #
####################################
solver_settings = {
    "iterations": 1400,
    "batch_size": None,
    "primal_solver": lambda p: torch.optim.Adam(p, lr=0.2),
    "dual_solver": lambda p: torch.optim.Adam(p, lr=0.01),
}
solver = csl.PrimalDual(solver_settings)

solutions = {}

for key, problem in problems.items():
    solver.reset()
    solver.solve(problem)
    solver.plot()  # displays diagnostic plots

    lambdas = problem.lambdas
    solver_state = solver.state_dict
    solutions[key] = {
        "model": problem.model,
        "lambdas": problem.lambdas,
        "solver_state": solver.state_dict,
    }


# ####################################
# # TESTING                          #
# ####################################
def accuracy(pred, y):
    correct = (pred == y).sum().item()
    return correct / pred.shape[0]


def d_l_hat(yhat, prob_feat):
    yhat = yhat.numpy()
    prob_feat = prob_feat.numpy()
    return np.sum((yhat - np.mean(yhat)) * (prob_feat - np.mean(prob_feat))) / np.sum(
        (prob_feat - np.mean(prob_feat)) ** 2
    )


def exp_condtl_cov(y_pred, prob_feat, true_feat):
    y_pred = y_pred.numpy()
    prob_feat = prob_feat.numpy()
    true_feat = true_feat.numpy()
    p_1 = np.mean(true_feat == 1)
    Y_hat_1 = y_pred[true_feat == 1]
    b_1 = prob_feat[true_feat == 1]
    b_bar_1 = np.mean(b_1)

    p_0 = np.mean(true_feat == 0)
    Y_hat_0 = y_pred[true_feat == 0]
    b_0 = prob_feat[true_feat == 0]
    b_bar_0 = np.mean(b_0)

    return p_1 * (1 / np.sum(true_feat == 1)) * np.sum(
        Y_hat_1 * (b_1 - b_bar_1)
    ) + p_0 * (1 / np.sum(true_feat == 0)) * np.sum(Y_hat_0 * (b_0 - b_bar_0))


for key, solution in solutions.items():
    print(f"Model: {key}")
    with torch.no_grad():
        x_test, y_test, prob_feat, true_feat = test_data[:]
        yhat = solution["model"].predict(x_test)
        acc_test = accuracy(yhat, y_test)
        # other metrics we want to evaluate here
        d_l_test = d_l_hat(yhat, prob_feat)
        exp_condtl_cov_test = exp_condtl_cov(yhat, prob_feat, true_feat)
        # let's check that our constraints are satisfied
        print(f"Accuracy: {acc_test:.4f}")
        print(f"D_l_hat for the test data: {d_l_test:.4f}")
        print(f"Exp_condtl_cov for the test data: {exp_condtl_cov_test:.4f}")
        print(f"Lambdas: {solution['lambdas']}")
