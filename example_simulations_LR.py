import xgboost as xgb
from sklearn.preprocessing import TargetEncoder
import optuna
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from functions import *

N_TRAIN_EXAMPLES = 5*10**4
DATA_SIZE = None
N_EPOCHS = 50

# Seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda")
    xgb_device = "cuda"
    xgb_tree_method = 'hist'
else:
    device = torch.device("cpu") 
    xgb_device = "cpu"
    xgb_tree_method = 'hist'

if not os.path.exists(here('output')):
    os.makedirs(here('output'))    

if not os.path.exists(here('output/model_hyperparameters')):
    os.makedirs(here('output/model_hyperparameters')) 

parser = ArgumentParser("Command line interface", formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--nsim", type=int, default=10, help="Number of simulations to run.") 
parser.add_argument("--ntrial", type=int, default=100, help="Number of trial for optuna hyperparameter tuning.")
parser.add_argument("--lr", type=int, default=0, help="Flag to run regular logistic regression.")
parser.add_argument("--lr_fair", type=int, default=0, help="Flag to run logistic regression with fairness penalty.")
parser.add_argument("--lambda_fair", type=float, default=0.1, help="Fairness penalty multiplier.")
parser.add_argument("--data_frac", type=float, default=1.0, help="Fraction of data to consider in the computation of the fairness penalty")
parser.add_argument("--name", type=str, default=datetime.today().strftime("%Y-%m-%d"), help="Additional name tag for saving results")
parser.add_argument("--data", type=str, default="fairjob.csv.gz", help="Dataset name in data/ folder.")
parser.add_argument("--batch", type=int, default=1024, help="Batch size.")
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
results_regular_df = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [
            ["LR"],
            list(range(args.nsim)),
        ],
        names=["model", "iteration"],
    ),
    columns=["NLLH", "DP", "UTILITY", "UTILITY_PRODUCT", "UTILITY_PRODUCT_FAIR", "AU-ROC", "AVG-P-SCORE"],
)

res_pred_df = pd.DataFrame(
    columns=[
        "model",
        "fairness_multiplier",
        "fairness_fraction",
        "iteration",
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

results_fair_df = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [
            ["LR"],
            [l2_fair_multiplier],
            [fair_fraction],  
            list(range(args.nsim)),
        ],
        names=["model", "fairness_multiplier", "fairness_fraction" ,"iteration"],
    ),
    columns=["NLLH", "DP", "UTILITY", "UTILITY_PRODUCT", "UTILITY_PRODUCT_FAIR", "AU-ROC", "AVG-P-SCORE"],
)

# Data loading
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

# Simulations
for sim in range(args.nsim):
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
        rank_test
    ) = train_test_split(X, y, protected_attribute, is_senior, displayrandom, rank)

    X_extended_train = torch.hstack(
        [displayrandom_train.unsqueeze(1), is_senior_train.unsqueeze(1), X_train, rank_train.unsqueeze(1)]
    )
    X_extended_test = torch.hstack(
        [displayrandom_test.unsqueeze(1), is_senior_test.unsqueeze(1), X_test, rank_test.unsqueeze(1)]
    )
    categorical_features_cardinalities_extended = {key+2: value for key,value in categorical_features_cardinalities.items()}
    categorical_features_cardinalities_extended[0] = 2 # cardinality for displayrandom
    categorical_features_cardinalities_extended[1] = 2 # cardinality for displayrandom
    
    impression_test = X_test[:,1]
    product_test = X_test[:,2]

    # Train data info
    print("Training data freq:")
    print(
        pd.crosstab(
            index=protected_attribute_train.detach().cpu().numpy(),
            columns=[y_train.detach().cpu().numpy(), is_senior_train.detach().cpu().numpy()],
            normalize="all",
            rownames=["protected attribute"],
            colnames=["clicks", "senior ads"],
            margins=True,
        )
    )
    print("\n")

    #############################
    #### LOGISTIC REGRESSION ####
    #############################
    if args.lr:
        print("\n RUNNING LOGISTIC REGRESSION \n")
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
            ) = train_test_split(X_train, y_train, protected_attribute_train, is_senior_train, displayrandom_train, rank_train)
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

            emb_size = trial.suggest_int('emb_size',4,5)
            learning_rate = trial.suggest_float('learning_rate',1e-4, 1e-2,log=True)
            weight_decay = trial.suggest_float('weight_decay',1e-6, 1e-4,log=True)
            scheduler_step_size = trial.suggest_int('scheduler_step_size',20,N_EPOCHS)
            scheduler_gamma = trial.suggest_float('scheduler_gamma',1e-2,1,log=True)

            lr = Learner(
                LogisticRegression(X_extended_train_train.shape[1], categorical_features_cardinalities_extended, emb_size),
                device=device,
                scheduler_step_size=scheduler_step_size,
                scheduler_gamma=scheduler_gamma,
                lr=learning_rate,
                weight_decay=weight_decay
            )
            for _ in range(N_EPOCHS):
                if _ * batch_size > N_TRAIN_EXAMPLES:
                    break
                lr.fit(
                    x = X_extended_train_train,
                    y = y_train_train,
                    a = protected_attribute_train_train,
                    batch_size = batch_size,
                )
                lr.scheduler_step()

                y_pred = lr(X_extended_val)
                intermediate_value = nn.CrossEntropyLoss()(y_pred, y_val).item()
                trial.report(intermediate_value, _)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            y_pred = lr(X_extended_val)
            return nn.CrossEntropyLoss()(y_pred, y_val).item()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        optimal = study.best_trial
        pd.DataFrame(optimal.params, index=[sim]).to_csv(
            here("output/model_hyperparameters/LR_opt_params" + args.name + ".csv"),
            mode='a', header=not here("output/model_hyperparameters/LR_opt_params" + args.name + ".csv")
        )

        # Fit on all training data
        lr = Learner(
            LogisticRegression(
                X_extended_train.shape[1],
                categorical_features_cardinalities_extended,
                optimal.params["emb_size"],
            ),
            device=device,
            scheduler_step_size=optimal.params['scheduler_step_size'],
            scheduler_gamma=optimal.params['scheduler_gamma'],
            lr=optimal.params["learning_rate"],
            weight_decay=optimal.params["weight_decay"],
        )
        for _ in range(N_EPOCHS):
            lr.fit(
                x=X_extended_train,
                y=y_train,
                a=protected_attribute_train,
                batch_size=batch_size,
            )
            lr.scheduler_step()

        res_pred_df = evaluate(
            res_pred_df,
            results_regular_df,
            "LR",
            None,
            None,
            sim,
            lr,
            X_extended_test,
            y_test,
            protected_attribute_test,
            is_senior_test,
            impression_test,
            displayrandom_test,
            product_test
        )
        prediction_stats(lr, X_extended_test, protected_attribute_test)

        # Intermediate save
        results_regular_df.to_csv(here('output/results_REGULAR_tuned' + args.name + '.csv'), mode='w+')
        res_pred_df.to_csv(here('output/pred' + args.name + '.csv'), mode='w+')

    
    
            
    ##################################
    #### FAIR LOGISTIC REGRESSION ####
    ##################################
    fair_indicator = torch.bernoulli( fair_fraction*torch.ones(size=y_train.shape) ).to(torch.int).to(device)
    
    if args.lr_fair:
        print(f"\n RUNNING FAIR LOGISTIC REGRESSION with lambda {l2_fair_multiplier} and frac {fair_fraction} \n")
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
            fair_indicator = torch.bernoulli( fair_fraction*torch.ones(size=y_train_train.shape) ).to(torch.int).to(device)

            emb_size = trial.suggest_int('emb_size',4,8)
            learning_rate = trial.suggest_float('learning_rate',1e-4, 1e-2,log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
            scheduler_step_size = trial.suggest_int('scheduler_step_size',20,N_EPOCHS)
            scheduler_gamma = trial.suggest_float('scheduler_gamma',1e-2,1,log=True)

            fair_lr = Learner(
                LogisticRegression(
                    X_extended_train_train.shape[1],
                    categorical_features_cardinalities_extended,
                    embedding_size=emb_size,
                ),
                device=device,
                scheduler_step_size=scheduler_step_size,
                scheduler_gamma=scheduler_gamma,
                basename="L2 FAIR",
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            for _ in range(N_EPOCHS):
                if _ * batch_size > N_TRAIN_EXAMPLES:
                    break
                fair_lr.fit(
                    x = X_extended_train_train,
                    y = y_train_train,
                    a = protected_attribute_train_train,
                    penalty_fun = l2_conditional_independence_penalty,
                    penalty_multiplier = l2_fair_multiplier,
                    fair_indicator = fair_indicator,
                    batch_size = batch_size,
                )
                fair_lr.scheduler_step()

                y_pred = fair_lr(X_extended_val)
                intermediate_value = nn.CrossEntropyLoss()(y_pred, y_val).item()
                trial.report(intermediate_value, _)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            y_pred = fair_lr(X_extended_val)

            return nn.CrossEntropyLoss()(y_pred, y_val).item()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        optimal = study.best_trial
        pd.DataFrame(
            optimal.params,
            index=pd.MultiIndex.from_product([[fair_fraction], [sim]], names=["fair_fraction", "sim"]),
        ).to_csv(
            here(
                "output/model_hyperparameters/FAIR_LR_opt_params"
                + args.name
                + ".csv"
            ),
            mode="a",
            header=not here(
                "output/model_hyperparameters/FAIR_LR_opt_params"
                + args.name
                + ".csv"
            ),
        )

        fair_lr = Learner(
            LogisticRegression(
                X_extended_train.shape[1], categorical_features_cardinalities_extended, embedding_size=optimal.params['emb_size'],
            ),
            device=device,
            scheduler_step_size=optimal.params['scheduler_step_size'],
            scheduler_gamma=optimal.params['scheduler_gamma'],
            basename="L2 FAIR",
            lr=optimal.params['learning_rate'],
            weight_decay=optimal.params['weight_decay'],
        )
        for _ in range(N_EPOCHS):
            fair_lr.fit(
                X_extended_train,
                y_train,
                protected_attribute_train,
                penalty_fun=l2_conditional_independence_penalty,
                penalty_multiplier=l2_fair_multiplier,
                batch_size=batch_size,
                fair_indicator = fair_indicator,
            )
            fair_lr.scheduler_step()

        res_pred_df = evaluate(
            res_pred_df,
            results_fair_df,
            "LR",
            l2_fair_multiplier,
            fair_fraction,
            sim,
            fair_lr,
            X_extended_test,
            y_test,
            protected_attribute_test,
            is_senior_test,
            impression_test,
            displayrandom_test,
            product_test
        )
        prediction_stats(fair_lr, X_extended_test, protected_attribute_test)

        # Intermediate save
        results_fair_df.to_csv(here('output/results_FAIR_tuned' + args.name + '.csv'), mode='w+')
        res_pred_df.to_csv(here('output/pred' + args.name + '.csv'), mode='w+')


# Saving final
results_fair_df.to_csv(here('output/results_FAIR_tuned' + args.name + '.csv'), mode='w+')
results_regular_df.to_csv(here('output/results_REGULAR_tuned' + args.name + '.csv'), mode='w+')
res_pred_df.to_csv(here('output/pred' + args.name + '.csv'), mode='w+')

