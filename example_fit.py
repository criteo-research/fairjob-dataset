import xgboost as xgb
from sklearn.preprocessing import TargetEncoder
import optuna
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from functions import *


N_TRAIN_EXAMPLES = 10**5
DATA_SIZE = None  # None for all data
sim = 0
N_EPOCHS = 50

# Seed for reproducibility
torch.manual_seed(43523)
np.random.seed(43523)

if torch.cuda.is_available():
    device = torch.device("cuda")
    xgb_device = "cuda"
    xgb_tree_method = "hist"
else:
    device = torch.device("cpu")
    xgb_device = "cpu"
    xgb_tree_method = "hist"

if not os.path.exists(here("output")):
    os.makedirs(here("output"))

if not os.path.exists(here("output/model_hyperparameters")):
    os.makedirs(here("output/model_hyperparameters"))

parser = ArgumentParser("Command line interface", formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--ntrial", type=int, default=100, help="Number of trial for optuna hyperparameter tuning.")
parser.add_argument("--dummy", type=int, default=0, help="Flag to run dummy classifier.")
parser.add_argument("--lr_fair", type=int, default=0, help="Flag to run logistic regression with fairness penalty.")
parser.add_argument("--xgb", type=int, default=0, help="Flag to run XGBoost.")
parser.add_argument("--lambda_fair", type=float, default=0.1, help="Fairness penalty multiplier.")
parser.add_argument("--data_frac", type=float, default=1.0, help="Fraction of data to consider in the computation of the fairness penalty")
parser.add_argument("--name", type=str, default=datetime.today().strftime("%Y-%m-%d"), help="Additional name tag for saving results")
parser.add_argument("--data", type=str, default="fairjob.csv.gz", help="Dataset name in data/ folder.")
parser.add_argument("--batch", type=int, default=1024, help="Batch size.")
parser.add_argument("--unfair", type=int, default=0, help="Flag for using the protected attribute as predictor.")
args = parser.parse_args()

batch_size = args.batch
l2_fair_multiplier = args.lambda_fair
fair_fraction = args.data_frac
n_trials = args.ntrial

# Print setup
print(args)
print('Device: ' + str(device))

# Name for saving results
args.name = (
    "_"
    + args.name
    + "_lambda"
    + str(args.lambda_fair)
    + "_frac"
    + str(args.data_frac)
)


# DataFrame for saving results
res_pred_df = pd.DataFrame(
    columns=[
        "model",
        "fairness_multiplier",
        "fairness_fraction",
        "obs_index",
        "prob_test",
        "y_test",
        "a_test",
        "s_test",
        "displayrandom_test",
        "impression_id_test",
        "product_id_test",
    ]
)

# Data loading and splitting
(
    X,
    y,
    protected_attribute,
    is_senior,
    displayrandom,
    rank,
    categorical_features_cardinalities,
) = load_data(DATA_SIZE, args.data)

X, y, protected_attribute, is_senior, displayrandom, rank = (
    Tensor(X.astype(np.float64)).to(device),
    Tensor(y).long().to(device),
    Tensor(protected_attribute).long().to(device),
    Tensor(is_senior).long().to(device),
    Tensor(displayrandom).long().to(device),
    Tensor(rank).long().to(device),
)

if args.unfair == 1:
    X = torch.hstack([X, protected_attribute.unsqueeze(dim=1)])
    args.name += "_unfair"

(
    X_train,
    X_test,
    y_train,
    y_test,
    protected_attribute_train,
    protected_attribute_test,
    is_senior_train,
    is_senior_test,
    displayrandom_train,
    displayrandom_test,
    rank_train,
    rank_test,
) = train_test_split(X, y, protected_attribute, is_senior, displayrandom, rank)

X_extended_train = torch.hstack(
    [
        displayrandom_train.unsqueeze(1),
        is_senior_train.unsqueeze(1),
        X_train,
        rank_train.unsqueeze(1),
    ]
)
X_extended_test = torch.hstack(
    [displayrandom_test.unsqueeze(1), is_senior_test.unsqueeze(1), X_test, rank_test.unsqueeze(1)]
)
categorical_features_cardinalities_extended = {
    key + 2: value for key, value in categorical_features_cardinalities.items()
}
categorical_features_cardinalities_extended[0] = 2  # cardinality for displayrandom
categorical_features_cardinalities_extended[1] = 2  # cardinality for is_senior

impression_test = X_test[:, 1]
product_test = X_test[:, 2]

fair_indicator = (
    torch.bernoulli(fair_fraction * torch.ones(size=y_train.shape)).to(torch.int).to(device)
)

data_train = JobDataset(X_extended_train, y_train, protected_attribute_train, fair_indicator)


###############
#### DUMMY ####
###############

if args.dummy:
    print("\n RUNNING DUMMY \n")
    dummy = Learner(DummyClassifier(), device=device)
    for _ in range(N_EPOCHS):
        dummy.fit(x=X_extended_train, y=y_train, a=protected_attribute_train, batch_size=batch_size)

    y_pred = dummy(X_extended_test)
    p = torch.softmax(y_pred, dim=1)
    res_pred_df = pd.concat(
        [
            res_pred_df,
            pd.DataFrame(
                {
                    "prob_test": p[:, 1].detach().cpu().numpy(),
                    "y_test": y_test.detach().cpu().numpy(),
                    "a_test": protected_attribute_test.detach().cpu().numpy(),
                    "s_test": is_senior_test.detach().cpu().numpy(),
                    "displayrandom_test": displayrandom_test.detach().cpu().numpy(),
                    "impression_id_test": impression_test.detach().cpu().numpy(),
                    "product_id_test": product_test.detach().cpu().numpy(),
                }
            )
            .reset_index(names="obs_index")
            .assign(model="Dummy", fairness_multiplier=None, fairness_fraction=None),
        ]
    )
    res_pred_df.to_csv(here("output/SINGLE_pred" + args.name + ".csv"), mode="w+")

    print(
        "DUMMY: NLLH: %.5f DP: %.5f UTILITY: %.5f UTILITY_P: %.5f UTILITY_P_FAIR: %.5f AU-ROC: %.5f AVG-P-SCORE: %.5f \n"
        % (
            nn.CrossEntropyLoss()(y_pred, y_test).item(),
            demographic_parity(
                p,
                protected_attribute_test,
                is_senior_test,
            ).item(),
            utility(
                p,
                y_test,
                protected_attribute_test,
                impression_test,
                displayrandom_test,
            ).item(),
            utility_product(
                p,
                y_test,
                protected_attribute_test,
                impression_test,
                displayrandom_test,
                product_test,
                unbiased_ratio=False,
            ).item(),
            utility_product(
                p,
                y_test,
                protected_attribute_test,
                impression_test,
                displayrandom_test,
                product_test,
                unbiased_ratio=True,
            ).item(),
            roc_auc_score(
                y_true=y_test.detach().cpu().numpy(),
                y_score=p[:, 1].detach().cpu().numpy(),
                average="macro",
            ),
            average_precision_score(
                y_true=y_test.detach().cpu().numpy(),
                y_score=p[:, 1].detach().cpu().numpy(),
                average="macro",
            ),
        )
    )


##################################
#### FAIR LOGISTIC REGRESSION ####
##################################
if args.lr_fair:
    print(
        f"\n RUNNING FAIR LOGISTIC REGRESSION with lambda {l2_fair_multiplier} and frac {fair_fraction} \n"
    )

    def objective(trial):
        (
            X_train_train,
            X_val,
            y_train_train,
            y_val,
            protected_attribute_train_train,
            protected_attribute_val,
            is_senior_train_train,
            is_senior_val,
            displayrandom_train_train,
            displayrandom_val,
            rank_train_train,
            rank_val,
        ) = train_test_split(
            X_train,
            y_train,
            protected_attribute_train,
            is_senior_train,
            displayrandom_train,
            rank_train,
        )
        X_extended_train_train = torch.hstack(
            [
                displayrandom_train_train.unsqueeze(1),
                is_senior_train_train.unsqueeze(1),
                X_train_train,
                rank_train_train.unsqueeze(1),
            ]
        )
        X_extended_val = torch.hstack(
            [
                displayrandom_val.unsqueeze(1),
                is_senior_val.unsqueeze(1),
                X_val,
                rank_val.unsqueeze(1),
            ]
        )
        fair_indicator = (
            torch.bernoulli(fair_fraction * torch.ones(size=y_train_train.shape))
            .to(torch.int)
            .to(device)
        )

        emb_size = trial.suggest_int("emb_size", 4, 8)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        fair_lr = Learner(
            LogisticRegression(
                X_extended_train_train.shape[1],
                categorical_features_cardinalities_extended,
                embedding_size=emb_size,
            ),
            device=device,
            basename="L2 FAIR",
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        for _ in range(N_EPOCHS):
            if _ * batch_size > N_TRAIN_EXAMPLES:
                break
            fair_lr.fit(
                x=X_extended_train_train,
                y=y_train_train,
                a=protected_attribute_train_train,
                penalty_fun=l2_conditional_independence_penalty,
                penalty_multiplier=l2_fair_multiplier,
                fair_indicator=fair_indicator,
                batch_size=batch_size,
            )
        y_pred = fair_lr(X_extended_val)

        return nn.CrossEntropyLoss()(y_pred, y_val).item()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    optimal = study.best_trial
    pd.DataFrame(
        optimal.params,
        index=pd.MultiIndex.from_product([[fair_fraction], [sim]], names=["fair_fraction", "sim"]),
    ).to_csv(
        here(
            "output/model_hyperparameters/SINGLE_FAIR_LR_W_opt_params" + args.name + ".csv"
        ),
        mode="a",
        header=not here(
            "output/model_hyperparameters/SINGLE_FAIR_LR_W_opt_params" + args.name + ".csv"
        ),
    )

    fair_lr = Learner(
        LogisticRegression(
            X_extended_train.shape[1],
            categorical_features_cardinalities_extended,
            embedding_size=optimal.params["emb_size"], 
        ),
        device=device,
        basename="L2 FAIR",
        lr=optimal.params["learning_rate"],
        weight_decay=optimal.params["weight_decay"],
    )
    for _ in range(N_EPOCHS):
        print("EPOCHS: " + str(_ + 1) + "/" + str(N_EPOCHS) + "\n")
        fair_lr.fit(
            x=X_extended_train,
            y=y_train,
            a=protected_attribute_train,
            penalty_fun=l2_conditional_independence_penalty,
            penalty_multiplier=l2_fair_multiplier,
            batch_size=batch_size,
            fair_indicator=fair_indicator,
        )
        with torch.no_grad():
            tmp = fair_lr(X_extended_train)
            tmp_test = fair_lr(X_extended_test)
        print(f"training NLLH : {nn.CrossEntropyLoss()(tmp,y_train).item()}")
        print(f"test NLLH : {nn.CrossEntropyLoss()(tmp_test,y_test).item()}")

    y_pred = fair_lr(X_extended_test)
    p = torch.softmax(y_pred, dim=1)
    res_pred_df = pd.concat(
        [
            res_pred_df,
            pd.DataFrame(
                {
                    "prob_test": p[:, 1].detach().cpu().numpy(),
                    "y_test": y_test.detach().cpu().numpy(),
                    "a_test": protected_attribute_test.detach().cpu().numpy(),
                    "s_test": is_senior_test.detach().cpu().numpy(),
                    "displayrandom_test": displayrandom_test.detach().cpu().numpy(),
                    "impression_id_test": impression_test.detach().cpu().numpy(),
                    "product_id_test": product_test.detach().cpu().numpy(),
                }
            )
            .reset_index(names="obs_index")
            .assign(
                model="LR", fairness_multiplier=l2_fair_multiplier, fairness_fraction=fair_fraction
            ),
        ]
    )
    res_pred_df.to_csv(here("output/SINGLE_pred" + args.name + ".csv"), mode="w+")


#################
#### XGBOOST ####
#################

if args.xgb:
    print("\n RUNNING XGBOOST \n")
    cat_cols = list(categorical_features_cardinalities_extended.keys())

    enc_auto = TargetEncoder(target_type="binary")
    enc_auto.fit(
        X_extended_train[:, cat_cols].detach().cpu().numpy(), y_train.detach().cpu().numpy()
    )

    def objective(trial):
        (
            X_train_train,
            X_val,
            y_train_train,
            y_val,
            protected_attribute_train_train,
            protected_attribute_val,
            is_senior_train_train,
            is_senior_val,
            displayrandom_train_train,
            displayrandom_val,
            rank_train_train,
            rank_val,
        ) = train_test_split(
            X_train,
            y_train,
            protected_attribute_train,
            is_senior_train,
            displayrandom_train,
            rank_train,
        )
        X_extended_train_train = torch.hstack(
            [displayrandom_train_train.unsqueeze(1), X_train_train, rank_train_train.unsqueeze(1)]
        )
        X_extended_val = torch.hstack(
            [displayrandom_val.unsqueeze(1), X_val, rank_val.unsqueeze(1)]
        )

        max_depth = trial.suggest_int("max_depth", 3, 10)
        min_child_weight = trial.suggest_float("min_child_weight", 0.0001, 100, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 1)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 1, log=True)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1)
        reg_lambda = trial.suggest_float("reg_lambda", 0.1, 10, log=True)
        gamma = trial.suggest_float("gamma", 0.001, 100, log=True)

        scale_pos_weight = (
            (torch.sum(y_train_train < 1) / torch.sum(y_train_train > 0)).detach().cpu().numpy()
        )

        X_xgb_train = np.hstack(
            [
                enc_auto.transform(X_extended_train_train[:, cat_cols].detach().cpu().numpy()),
                X_extended_train_train[:, (max(cat_cols) + 1) :].detach().cpu().numpy(),
            ]
        )
        X_xgb_val = np.hstack(
            [
                enc_auto.transform(X_extended_val[:, cat_cols].detach().cpu().numpy()),
                X_extended_val[:, (max(cat_cols) + 1) :].detach().cpu().numpy(),
            ]
        )

        model_xgb = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            tree_method=xgb_tree_method,
            device=xgb_device,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            learning_rate=learning_rate,
            gamma=gamma,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
        )
        model_xgb.fit(X_xgb_train, y_train_train.detach().cpu().numpy())
        prob_val = model_xgb.predict_proba(X_xgb_val)
        return log_loss(y_val.detach().cpu().numpy(), prob_val)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    optimal = study.best_trial
    pd.DataFrame(optimal.params, index=[sim]).to_csv(
        here("output/model_hyperparameters/XGBoost_opt_params" + args.name + ".csv"),
        mode="a",
        header=not here(
            "output/model_hyperparameters/XGBoost_opt_params" + args.name + ".csv"
        ),
    )
    scale_pos_weight = (torch.sum(y_train < 1) / torch.sum(y_train > 0)).detach().cpu().numpy()
    X_xgb_train = np.hstack(
        [
            enc_auto.transform(X_extended_train[:, cat_cols].detach().cpu().numpy()),
            X_extended_train[:, (max(cat_cols) + 1) :].detach().cpu().numpy(),
        ]
    )
    X_xgb_test = np.hstack(
        [
            enc_auto.transform(X_extended_test[:, cat_cols].detach().cpu().numpy()),
            X_extended_test[:, (max(cat_cols) + 1) :].detach().cpu().numpy(),
        ]
    )
    model_xgb = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        tree_method=xgb_tree_method,
        device=xgb_device,
        max_depth=optimal.params["max_depth"],
        min_child_weight=optimal.params["min_child_weight"],
        gamma=optimal.params["gamma"],
        subsample=optimal.params["subsample"],
        learning_rate=optimal.params["learning_rate"],
        colsample_bytree=optimal.params["colsample_bytree"],
        reg_lambda=optimal.params["reg_lambda"],
    )
    model_xgb.fit(X_extended_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())

    if args.unfair == 0:
        model_xgb.save_model(here("output/XGB_single_fit.model"))
    else:
        model_xgb.save_model(here("output/XGB_single_fit_unfair.model"))

    prob_test = model_xgb.predict_proba(X_extended_test.detach().cpu().numpy())

    res_pred_df = pd.concat(
        [
            res_pred_df,
            pd.DataFrame(
                {
                    "prob_test": prob_test[:, 1],
                    "y_test": y_test.detach().cpu().numpy(),
                    "a_test": protected_attribute_test.detach().cpu().numpy(),
                    "s_test": is_senior_test.detach().cpu().numpy(),
                    "displayrandom_test": displayrandom_test.detach().cpu().numpy(),
                    "impression_id_test": impression_test.detach().cpu().numpy(),
                    "product_id_test": product_test.detach().cpu().numpy(),
                }
            )
            .reset_index(names="obs_index")
            .assign(model="XGB", fairness_multiplier=None, fairness_fraction=None),
        ]
    )
    res_pred_df.to_csv(here("output/SINGLE_pred" + args.name + ".csv"), mode="w+")

    print(
        "\nXGBoost: NLLH: %.5f DP: %.5f UTILITY: %.5f UTILITY_P: %.5f UTILITY_P_FAIR: %.5f AU-ROC: %.5f AVG-P-SCORE: %.5f \n"
        % (
            log_loss(y_test.detach().cpu().numpy(), prob_test),
            demographic_parity(
                Tensor(prob_test.astype(np.float64)).to(device),
                protected_attribute_test,
                is_senior_test,
            ).item(),
            utility(
                Tensor(prob_test.astype(np.float64)).to(device),
                y_test,
                protected_attribute_test,
                impression_test,
                displayrandom_test,
            ).item(),
            utility_product(
                Tensor(prob_test.astype(np.float64)).to(device),
                y_test,
                protected_attribute_test,
                impression_test,
                displayrandom_test,
                product_test,
                unbiased_ratio=False,
            ).item(),
            utility_product(
                Tensor(prob_test.astype(np.float64)).to(device),
                y_test,
                protected_attribute_test,
                impression_test,
                displayrandom_test,
                product_test,
                unbiased_ratio=True,
            ).item(),
            roc_auc_score(
                y_true=y_test.detach().cpu().numpy(),
                y_score=prob_test[:, 1],
                average="macro",
            ),
            average_precision_score(
                y_true=y_test.detach().cpu().numpy(),
                y_score=prob_test[:, 1],
                average="macro",
            ),
        )
    )


# Final save
res_pred_df.to_csv(here("output/SINGLE_pred" + args.name + ".csv"), mode="w+")
