import numpy as np
import torch
from pyprojroot.here import here
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score
from pop_constants import *

# ----------------------------------------------------------------------------------------------------------------------
# Data utilities
# ----------------------------------------------------------------------------------------------------------------------

def load_data(
    nrows: int = None, 
    filename: str = "fairjob.csv.gz"
):
    """
    Data loader for FairJob dataset.
    
    Args:
        nrows (int): How many rows to load for np.loadtxt. Defaults to None.
        filename (str): Name of data file located in data/. Defaults to "fairjob.csv.gz".

    Returns:
        X (numpy.ndarray): (n_data, n_features)
        click (np.ndarray): (n_data,)
        protected_attribute (numpy.ndarray): (n_data,)
        senior (numpy.ndarray): (n_data,)
        displayrandom (numpy.ndarray): (n_data,)
        rank (numpy.ndarray): (n_data,)
        categorical_features_cardinalities (dict): categorical features cardinalities
    """
    
    data = np.loadtxt(here("data/" + filename), skiprows=1, delimiter=",", max_rows=nrows)
    click_idx = 0
    protected_attribute_idx = 1
    senior_idx = 2
    displayrandom_idx = 3
    rank_idx = 4
    X = data[:, 5:]
    n_cat_cols = np.sum(X.mean(axis=0) > 1e-1)
    categorical_features_idx = np.arange(0, n_cat_cols)
    categorical_features_cardinalities = dict()

    # counting unique tokens per categorical variable
    # and renumbering them from 0
    for dim in range(len(categorical_features_idx)):
        renumber_dict = dict()
        values = np.unique(X[:, dim])
        categorical_features_cardinalities[dim] = len(values)
        for i, v in enumerate(values):
            renumber_dict[v] = i
        for i in range(len(X)):
            X[i, dim] = renumber_dict[X[i, dim]]

    return (
        X,
        data[:, click_idx],
        data[:, protected_attribute_idx],
        data[:, senior_idx],
        data[:, displayrandom_idx],
        data[:, rank_idx],
        categorical_features_cardinalities,
    )

class JobDataset(Dataset):
    """ 
    Class collecting the variables of the data necessary for batch generation.
    """

    def __init__(self, X, y, protected_attribute, fair_indicator):
        self.X = X
        self.y = y
        self.protected_attribute = protected_attribute
        if fair_indicator is None:
            self.fair_indicator = torch.ones_like(y).to(torch.int).to(y.device)
        else:
            self.fair_indicator = fair_indicator

    def __len__(self):
        return len(self.y)

    def get_target(self):
        return self.y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.protected_attribute[idx], self.fair_indicator[idx]

def batch_loader(dataset: JobDataset, batch_size: int):
    """ 
    Factory for batch data loader with weighted random sampling for addressing class imbalance.

    Args:
        dataset (JobDataset): Training data 
        batch_size (int): Batch size

    Returns:
        DataLoader
    """
    
    class_counts = torch.bincount(dataset.get_target())
    class_weights = 1.0 / class_counts.float()
    # Create weights for each sample in the dataset
    sample_weights = class_weights[dataset.get_target()]
    sample_weights = sample_weights.detach().cpu().numpy()

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)


def train_test_split(
    X, y, protected_attribute, is_senior, displayrandom, rank, train_fraction: float = 0.8
):
    """ 
    Random split of data in training and test sets.

    Args:
        X: Features
        y: Target labels
        protected_attribute: Protected attribute
        is_senior: Flag for senior ads.
        displayrandom: Flag for ads with random display order in the banner.
        rank: Ads rank in the banner.
        train_fraction (float): Fraction for training data. Defaults to 0.8.

    Returns:
        X_train,
        X_test,
        y_train,
        y_test,
        protected_attribute_train,
        protected_attribute_test,
        is_senior_train,
        is_senior_test,
        displayrandom_train,
        displayrandom_test
        rank_train,
        rank_test
    """
    
    cut = int(len(X) * train_fraction * 100 // 100)
    X_train = X[:cut, :]
    X_test = X[cut:, :]
    y_train = y[:cut]
    y_test = y[cut:]
    protected_attribute_train = protected_attribute[:cut]
    protected_attribute_test = protected_attribute[cut:]
    is_senior_train = is_senior[:cut]
    is_senior_test = is_senior[cut:]
    displayrandom_train = displayrandom[:cut]
    displayrandom_test = displayrandom[cut:]
    rank_train = rank[:cut]
    rank_test = rank[cut:]
    return (
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
    )


# ----------------------------------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------------------------------

class DummyClassifier(nn.Module):
    """
    Classifier based on single threshold for positive class probability.
    """

    def __init__(self, start_p: float = 0.005):
        super().__init__()
        self.bias = nn.Parameter(Tensor([1 - start_p, start_p]))
        self.bias.requires_grad = True
        self.register_buffer("zero_const", torch.zeros((1, 2)))

    def forward(self, x):
        return self.zero_const.repeat(x.shape[0], 1) + self.bias

    def __str__(self):
        return "Dummy"


class MixedEmbedding(nn.Module):
    """
    Layer for embedding categorical and continuous features for Logistic Regression.
    """

    def __init__(
        self,
        input_dim: int,
        categorical_features_cardinalities: dict,
        embedding_size: int = 10,
    ):
        """
        Args:
            input_dim (int): dimension of input features
            categorical_features_cardinalities (dict): cardinalities (values) for each categorical feature (column index in the data as key)
            embedding_size (int): Output size of the mixed embedding layer. Defaults to 10.
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_size = embedding_size
        self.categorical_features_idx = np.array(
            [_ for _ in categorical_features_cardinalities.keys()]
        )
        self.numerical_features_idx = np.array(
            [_ for _ in range(input_dim) if _ not in self.categorical_features_idx]
        )
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=categorical_features_cardinalities[cat_feature_idx],
                    embedding_dim=min(
                        embedding_size,
                        categorical_features_cardinalities[cat_feature_idx],
                    ),
                )
                for cat_feature_idx in self.categorical_features_idx
            ]
        )
        self.output_dimension = len(self.numerical_features_idx) + sum(
            [e.embedding_dim for e in self.embeddings]
        )

    def forward(self, x):
        catx = x[:, self.categorical_features_idx].int()
        numx = x[:, self.numerical_features_idx]
        embeddedx = torch.hstack(
            [self.embeddings[_](catx[:, _]) for _ in self.categorical_features_idx]
        )
        embeddedx = embeddedx.view(-1, self.output_dimension - len(self.numerical_features_idx))
        return torch.hstack([embeddedx, numx])

    def to(self, device):
        self.embeddings.to(device)


class LogisticRegression(nn.Module):
    """
    Logistic regression classifier.
    """

    def __init__(
        self,
        input_dim: int,
        categorical_features_cardinalities: dict,
        embedding_size: int,
    ):
        """
        Args:
            input_dim (int): dimension of input features
            categorical_features_cardinalities (dict): cardinalities (values) for each categorical feature (column index in the data as key)
            embedding_size (int): Output size of the mixed embedding layer.
        """
        super().__init__()
        self.mixed_embedding_layer = MixedEmbedding(
            input_dim,
            categorical_features_cardinalities,
            embedding_size=embedding_size,
        )
        self.weights = nn.Linear(
            in_features=self.mixed_embedding_layer.output_dimension, out_features=2
        )

    def forward(self, x):
        xx = self.mixed_embedding_layer(x)
        return self.weights(xx)

    def __str__(self):
        return "LR"

    def to(self, device):
        self.mixed_embedding_layer.to(device)
        self.weights.to(device)


# ----------------------------------------------------------------------------------------------------------------------
# Learning routines
# ----------------------------------------------------------------------------------------------------------------------

def l2_conditional_independence_penalty(
    y_hat: Tensor, y_true: Tensor, protected_attribute: Tensor, fair_indicator: Tensor, *kwars
):
    """
    Implementation of the fairness penalty based on Bechavod & Ligett - https://arxiv.org/abs/1707.00044.
    Differently from the original proposal, it uses the squared discrepancies between
    the unconditional values of FPR, FNR, TPR, TNR and their conditional values, for both protected attribute labels.

    Args:
        y_hat (Tensor): Predicted probabilities for the two target labels. Dimensions: (n_data, 2)
        y_true (Tensor): True labels. Dimensions: (n_data,)
        protected_attribute (Tensor): Protected attributed values. Dimensions: (n_data,)
        fair_indicator (Tensor): Indicator for observations to be included in the penalty computation. Dimensions: (n_data,)

    Returns:
        Value of the fairness penalty
    """
    
    assert y_hat.shape[1] == 2
    sum_of_squares = Tensor([0.0]).to(y_hat.device)

    y_hat = y_hat[fair_indicator, :]
    y_true = y_true[fair_indicator]
    protected_attribute = protected_attribute[fair_indicator]
    for y_hat_dim in (0, 1):
        y_hat_col = y_hat[:, y_hat_dim]
        for y_dim in (0, 1):
            if torch.sum(y_true == y_dim) == 0.0:  # no such label
                continue
            y_hat_avg = torch.mean(y_hat_col[y_true == y_dim])
            for a_dim in (0, 1):
                if (
                    torch.sum((y_true == y_dim) & (protected_attribute == a_dim)) == 0.0
                ):  # no such label given attribute
                    y_hat_cond_a_avg = torch.tensor(0.0)
                else:
                    y_hat_cond_a_avg = torch.mean(
                        y_hat_col[(y_true == y_dim) & (protected_attribute == a_dim)]
                    )
                sum_of_squares += (y_hat_avg - y_hat_cond_a_avg) ** 2
    penalty = torch.sqrt(sum_of_squares)[0]
    return penalty


class Learner(object):
    """
    General class for learner.
    """

    def __init__(
        self,
        base_model,
        device="cuda",
        basename: str = "REGULAR",
        scheduler_step_size=30,
        scheduler_gamma=0.1,
        **optimizer_options
    ):
        """
        Args:
            base_model: Base model class  (e.g., Dummy, LogisticRegression)
            device (str): Device for PyTorch. Defaults to "cuda".
            basename (str): Model name. Defaults to "REGULAR".
            scheduler_step_size (int): Step size parameter for learning rate scheduler. Defaults to 30.
            scheduler_gamma (float): Gamma parameter for learning rate scheduler. Defaults to 0.1.
        """ 
        
        self.model = base_model
        self.model.ps_init = nn.Parameter(torch.randn(1), requires_grad=False)
        self.model.ps = nn.Parameter(self.model.ps_init.data, requires_grad=True)
        self.basename = basename
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_options)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
        )
        self.device = device
        self.model.to(device)

    def __str__(self):
        return self.basename + " " + str(self.model).split("(")[0]

    def fit(
        self,
        x: Tensor,
        y: Tensor,
        a: Tensor,
        batch_size=1024,
        penalty_fun=None,
        penalty_multiplier: float = 0.1,
        fair_indicator: Tensor = None,
    ):
        """
        Fit function (single epoch).

        Args:
            x (Tensor): Training features
            y (Tensor): Training target labels
            a (Tensor): Training protected attribute
            batch_size (int): Batch size. Defaults to 1024.
            penalty_fun : Additional penalty function to include fairness penalty. Defaults to None.
            penalty_multiplier (float): Multiplier for additional penalty function. Defaults to 0.1.
            fair_indicator (Tensor):  Indicator for observations to be included in the penalty computation. Defaults to None.

        Returns:
            Loss value
        """
        
        loss_value = 0

        data_train = JobDataset(x, y, a, fair_indicator)
        for batch in batch_loader(data_train, batch_size=batch_size):
            x_b, y_b, a_b, fair_indicator_b = batch
            o_b = x_b[:, 1]
            self.optimizer.zero_grad()
            outputs_b = self.model(x_b)

            loss_b = self.loss(outputs_b, y_b)

            if penalty_fun is not None:
                probas_b = torch.softmax(outputs_b, dim=1)
                assert torch.max(probas_b) <= 1.0, torch.max(probas_b)
                assert torch.min(probas_b) >= 0.0, torch.min(probas_b)
                loss_b += penalty_multiplier * penalty_fun(
                    y_hat=probas_b,
                    y_true=y_b,
                    protected_attribute=a_b,
                    fair_indicator=fair_indicator_b,
                )

            loss_b.backward()
            loss_value += loss_b.detach().cpu().numpy()
            self.optimizer.step()

        return loss_value

    def scheduler_step(self):
        self.scheduler.step()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_ps(self, *args, **kwargs):
        return self.model.ps.data

    def get_ps_init(self, *args, **kwargs):
        return self.model.ps_init.data

    def to(self, device: str):
        self.device = device
        self.model.to(device)
        self.loss.to(device)

# ----------------------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------------------


def demographic_parity(predictions, protected_attribute, scope):
    """
    Function to compute demographic parity with respect to scope.

    Args:
        predictions: model predictions
        protected_attribute: protected attribute values
        scope: indicator for scope

    Returns:
       Demographic parity
    """

    res = 0.0
    for col in (0, 1):
        res += torch.abs(
            torch.mean(predictions[(protected_attribute > 0) & (scope > 0)][:, col])
            - torch.mean(predictions[(protected_attribute <= 0) & (scope > 0)][:, col])
        )
    return res / 2.0


def utility(
    y_pred: Tensor,
    y: Tensor,
    protected_attribute: Tensor,
    impressions: Tensor,
    displayrandom: Tensor,
):
    """
    Click utility function.

    Args:
        y_pred (Tensor): Predicted probabilities for the two target labels. Dimensions (n_data, 2)
        y (Tensor): Target labels. Dimensions (n_data,)
        protected_attribute (Tensor): Protected attribute. Dimensions (n_data,)
        impressions (Tensor): Impression_id. Dimensions (n_data,)
        displayrandom (Tensor): Flag for ads with random display order in the banner. Dimensions (n_data,)

    Returns:
        Value of the utility
    """

    # Higher is better
    res = 0.0
    click_col = 1
    # Subselect only data with randomized display
    mask = (displayrandom > 0)
    y_pred = y_pred[mask]
    y = y[mask]
    protected_attribute = protected_attribute[mask]
    impressions = impressions[mask]

    for impression in torch.unique(impressions):
        y_ranked = (torch.argsort(y_pred[impressions == impression,click_col]) + 1).to(torch.float)  
        y_clicked = y[impressions == impression] 
        click_rank_impression = torch.mean(y_ranked * y_clicked) 

        res += click_rank_impression

    return res / (torch.unique(impressions)).size(dim=0)


def utility_product(
    y_pred, y, protected_attribute, impressions, displayrandom, product, unbiased_ratio=False
):
    """
    Product utility function.

    Args:
        y_pred (Tensor): Predicted probabilities for the two target labels. Dimensions (n_data, 2)
        y (Tensor): Target labels. Dimensions (n_data,)
        protected_attribute (Tensor): Protected attribute. Dimensions (n_data,)
        impressions (Tensor): Impression_id. Dimensions (n_data,)
        displayrandom (Tensor): Flag for ads with random display order in the banner. Dimensions (n_data,)
        unbiased_ratio (bool): True if the utility should be corrected with the data-population ratio of the protected attribute. Defaults to False.

    Returns:
        Value of the utility
    """
    
    res = 0.0
    click_col = 1
    # Subselect only data with randomized display
    mask = (displayrandom > 0)
    y_pred = y_pred[mask]
    y = y[mask]
    protected_attribute = protected_attribute[mask]
    impressions = impressions[mask]
    product = product[mask]

    # Create rank probs for each impression
    y_ranked = torch.zeros_like(y,dtype=torch.float)
    for impression in torch.unique(impressions):
        y_ranked[impressions == impression] = (torch.argsort(y_pred[impressions == impression,click_col]) + 1).to(torch.float)  

    if unbiased_ratio:
        ratio = (protected_attribute == 0) * FEMALE_RATIO + \
            (protected_attribute == 1) * MALE_RATIO
    else:
        ratio = torch.ones_like(protected_attribute, dtype=torch.float)

    for prod in torch.unique(product):
        prod_mask = (product == prod)
        for impression in torch.unique(impressions[prod_mask]):
            size_impressions_prod = torch.unique(impressions[prod_mask]).shape[0]
            impression_prod_mask = (impressions == impression) & prod_mask # Not really necessary, each impression is for a specific product
            y_ranked_prod = y_ranked[impression_prod_mask]
            y_clicked_prod = y[impression_prod_mask]

            click_rank_impression = (y_ranked_prod * ratio[impression_prod_mask]).dot(y_clicked_prod.to(torch.float))
            res += click_rank_impression/size_impressions_prod

    return res/ torch.unique(product).shape[0]


def evaluate(
    res_pred_df: pd.DataFrame,
    results_df: pd.DataFrame,
    name: str,
    l2_fair: float,
    fair_frac: float,
    sim: int,
    model,
    X_test: Tensor,
    y_test: Tensor,
    protected_attribute_test: Tensor,
    is_senior_test: Tensor,
    impression_test: Tensor,
    displayrandom_test: Tensor,
    product_test: Tensor,
):
    """
    Function for computing model evaluation metrics and save results.

    Args:
        res_pred_df (pd.DataFrame): DataFrame for saving model predictions.
        results_df (pd.DataFrame): DataFrame for saving model metrics.
        name (str): Model name as used in DataFrame index.
        l2_fair (float): Fairness penalty multiplier. Use None if without penalty.
        fair_frac (float): Fraction of data used to compute the fairness penalty. Use None if without penalty.
        sim (int): Simulation index.
        model: Model to use for predictions.
        X_test (Tensor): Test features.
        y_test (Tensor): Test target labels.
        protected_attribute_test (Tensor): Test protected attribute.
        is_senior_test (Tensor): Test senior ads indicator.
        impression_test (Tensor): Test impression id.
        displayrandom_test (Tensor): Test displayrandom.
        product_test (Tensor): Test product id.

    Returns:
        res_pred_df: updated with new values. Note that results_df is modified in-place.
    """
    
    with torch.no_grad():
        y_pred = model(X_test)
        p = torch.softmax(y_pred, dim=1)
    if l2_fair is not None:
        # We save fair models
        rows_index = (name,l2_fair,fair_frac,sim)
    else:
        # We save regular models
        rows_index = (name,sim)

    # Saving prediction
    res_pred_df = pd.concat(
        [
            res_pred_df,
            pd.DataFrame(
                {
                    "prob_test": p[:,1].detach().cpu().numpy(),
                    "y_test": y_test.detach().cpu().numpy(),
                    "a_test": protected_attribute_test.detach().cpu().numpy(),
                    "s_test": is_senior_test.detach().cpu().numpy(),
                    "displayrandom_test": displayrandom_test.detach().cpu().numpy(),
                    "impression_id_test": impression_test.detach().cpu().numpy(),
                    "product_id_test": product_test.detach().cpu().numpy(),
                }
            ).reset_index(names="obs_index").assign(model=name,
                                                    fairness_multiplier=l2_fair,
                                                    fairness_fraction=fair_frac,
                                                    iteration=sim),
        ]
    )

    results_df.loc[rows_index] = {
        "NLLH": nn.CrossEntropyLoss()(y_pred, y_test).item(),
        "DP": demographic_parity(p, protected_attribute_test, is_senior_test).item(),
        "UTILITY": utility(
            p, y_test, protected_attribute_test, impression_test, displayrandom_test
        ).item(),
        "UTILITY_PRODUCT": utility_product(
            p, y_test, protected_attribute_test, impression_test, displayrandom_test, product_test, unbiased_ratio=False
        ).item(),
        "UTILITY_PRODUCT_FAIR": utility_product(
            p, y_test, protected_attribute_test, impression_test, displayrandom_test, product_test, unbiased_ratio=True
        ).item(),
        "AU-ROC": roc_auc_score(y_true=y_test.detach().cpu().numpy(),
                                y_score=p[:,1].detach().cpu().numpy(),
                                average='macro'),
        "AVG-P-SCORE": average_precision_score(y_true=y_test.detach().cpu().numpy(),
                                y_score=p[:,1].detach().cpu().numpy(),
                                average='macro'),
    }
    print(
        "(%20s) NLLH: %.5f DP: %.5f UTILITY: %.5f UTILITY_P: %.5f UTILITY_P_FAIR: %.5f AU-ROC: %.5f AVG-P-SCORE: %.5f"
        % (
            model,
            results_df.loc[rows_index,'NLLH'],
            results_df.loc[rows_index,'DP'],
            results_df.loc[rows_index,'UTILITY'],
            results_df.loc[rows_index,'UTILITY_PRODUCT'],
            results_df.loc[rows_index,'UTILITY_PRODUCT_FAIR'],
            results_df.loc[rows_index,'AU-ROC'],
            results_df.loc[rows_index,'AVG-P-SCORE']
        )
    )
    return res_pred_df


def prediction_stats(model, X_test: Tensor, protected_attribute_test: Tensor):
    """
    Prints the prediction statistics with respect to protected attribute.

    Args:
        model: Model used for prediction
        X_test (Tensor): Test features.
        protected_attribute_test (_type_): Test protected attribute.
    """
    
    with torch.no_grad():
        y_pred = model(X_test)   
    res_fit_df = pd.DataFrame({'attribute': protected_attribute_test.detach().cpu().numpy(), 
                               0: torch.softmax(y_pred,dim=1).detach().cpu().numpy()[:,0], 
                               1: torch.softmax(y_pred,dim=1).detach().cpu().numpy()[:,1]})
    print(res_fit_df.set_index('attribute').groupby('attribute').mean())
    print(torch.softmax(y_pred,dim=1).detach().cpu().numpy()[0,:])
    print('\n')
