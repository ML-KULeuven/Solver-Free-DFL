import numpy as np
import pyepo
import pyepo.data.shortestpath
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data import polynomial_mapping as polynomial_mapping_adapted
from data.dataset_augmented import optDatasetAugmented, collate_fn_2
from data.generate_california_house_prices_mapping import (
    generate_california_house_prices_mapping,
)
from optimization_problems.knapsack import knapsackModel
from optimization_problems.random_lp import RandomLP
from optimization_problems.shortest_path import shortestPathModel
from utils.misc import set_seed


def create_datasets(params):
    """
    Create training and test datasets for the specified optimization
    problem.

    Args:
        params (dict): Dictionary containing all parameters including:

            - optimization_problem (str): Type of optimization problem
              ('random_lp', 'shortest_path', 'knapsack',
              'california_house_price_knapsack')
            - num_data (int): Number of data points to generate
            - num_feat (int): Number of features
            - deg (int): Degree parameter for data generation
            - e (float): Noise parameter for data generation
            - seed (int): Random seed
            - Additional parameters specific to each optimization problem

    Returns:
        tuple: (
            optmodel,
            mm,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            adj_vert_computation_time,
            dirs_computation_time
        )
    """

    benchmark = params['benchmark']
    num_data = params['num_data']
    seed = params['seed']
    set_seed(seed)

    if benchmark == "random_lp":
        num_vars = params['num_vars']
        num_constrs = params['num_constrs']
        use_new_generation_method = params['use_new_generation_method']
        add_variable_upper_bounds = params['add_variable_upper_bounds']

        x, c = polynomial_mapping_adapted.genData(num_data + 1500,
                                                  params['num_feat'],
                                                  num_vars,
                                                  params['deg'],
                                                  params['e'],
                                                  seed=seed)
        optmodel = RandomLP(num_vars=num_vars, num_constrs=num_constrs,
                            use_new_generation_method=use_new_generation_method,
                            add_variable_upper_bounds=add_variable_upper_bounds,
                            seed=seed)

    elif benchmark == "shortest_path":
        grid_width = params['grid_width']
        grid = (grid_width, grid_width)

        x, c = pyepo.data.shortestpath.genData(num_data + 1500, params['num_feat'], grid, params['deg'], params['e'],
                                               seed=seed)
        optmodel = shortestPathModel(grid=grid)

    elif benchmark == "knapsack":
        num_items = params['num_items']
        num_vars = num_items
        num_constrs = params['dims']

        weights, x, c = pyepo.data.knapsack.genData(num_data + 1500, params['num_feat'], num_vars, num_constrs,
                                                    params['deg'], params['e'], seed)
        capacities = np.array(0.5 * np.sum(weights, axis=1))
        optmodel = knapsackModel(weights=weights, capacities=capacities)

    elif benchmark == "california_house_price_knapsack":
        num_items = params['num_items']
        num_vars = num_items

        x_train, c_train, x_val, c_val, x_test, c_test = (
            generate_california_house_prices_mapping(n_train=num_data,
                                                     n_test=250,
                                                     num_houses_per_instance=num_vars))

        weights = np.random.randint(1, 10, size=(params["dims"], num_items))
        capacities = np.array(0.1 * np.sum(weights, axis=1))
        optmodel = knapsackModel(weights=weights, capacities=capacities)

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    mm = -1 * optmodel.modelSense

    # Split data
    if benchmark != "california_house_price_knapsack":  # Data is already split for california_house_price_knapsack
        # First split to separate test set
        x_temp, x_test, c_temp, c_test = train_test_split(x, c, test_size=1000, random_state=100 + seed)
        # Then split remaining data into train and validation
        x_train, x_val, c_train, c_val = train_test_split(x_temp, c_temp, test_size=500, random_state=200 + seed)

    # Build data sets and data loaders
    train_dataset = optDatasetAugmented(optmodel, x_train, c_train)
    adj_vert_computation_time = train_dataset.adj_vert_computation_time

    val_dataset = pyepo.data.dataset.optDataset(optmodel, x_val, c_val)
    test_dataset = pyepo.data.dataset.optDataset(optmodel, x_test, c_test)
    train_dataloader = DataLoader(train_dataset, batch_size=params["batch_size"], collate_fn=collate_fn_2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

    return optmodel, mm, train_dataloader, val_dataloader, test_dataloader, adj_vert_computation_time
