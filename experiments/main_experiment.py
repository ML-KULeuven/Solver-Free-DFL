import time
import json
import argparse
import os

from architectures.model_factory import create_model
from data.dataset_generator import create_datasets
from utils.regret import regret
from plotting.plot_results import plot_results
from utils.train_model import train_model

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_random_lp.json',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Create timestamp directory for results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = f"results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save config file copy
    config = load_config(args.config)
    with open(f"{results_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=4)

    # Extract parameters from config + setup
    params = {k: v for k, v in config.items() if k not in ["experiments", "seeds"]}
    print(f"\nRunning with parameters: {params}")
    experiments = config['experiments']
    seeds = config['seeds']
    num_epochs = params['num_epochs']
    log_every_n_epochs = params['log_every_n_epochs']
    methods = []
    train_regrets_all = []
    val_regrets_all = []
    test_regrets_all = []
    train_times_all = []

    # Generate all datasets first for each seed
    datasets_by_seed = {}
    for seed in seeds:
        params['seed'] = seed
        datasets_by_seed[seed] = create_datasets(params)

    # Run each method multiple times
    for experiment in experiments:
        methods.append(experiment["name"])
        method_train_regrets = []
        method_val_regrets = []
        method_test_regrets = []
        method_train_times = []

        for seed in seeds:
            # Reuse previously created datasets
            (optmodel, mm, train_dataloader, val_dataloader, test_dataloader, adj_vert_computation_time) =\
                datasets_by_seed[seed]

            num_feat = train_dataloader.dataset.feats.shape[-1]
            num_vars = train_dataloader.dataset.costs.shape[1]
            reg = create_model(params["model"], num_feat, num_vars)

            train_regret, val_regret, train_time = train_model(
                reg=reg,
                optmodel=optmodel,
                method=experiment["method"],
                num_epochs=num_epochs,
                lr=experiment["learning_rate"],
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                hyperparameters=experiment["hyperparameters"],
                seed=seed,
                adj_vert_computation_time=adj_vert_computation_time,
                log_every_n_epochs=log_every_n_epochs,
                mm=mm
            )

            test_regret = regret(reg, optmodel, test_dataloader)
            method_train_regrets.append(train_regret)
            method_val_regrets.append(val_regret)
            method_test_regrets.append(test_regret)
            method_train_times.append(train_time)

        train_regrets_all.append(method_train_regrets)
        val_regrets_all.append(method_val_regrets)
        test_regrets_all.append(method_test_regrets)
        train_times_all.append(method_train_times)

    # Save results in JSON format
    results_data = {
        'parameters': params,
        'methods': []
    }
    for i, method_name in enumerate(methods):
        method_data = {
            'name': method_name,
            'train_regrets': train_regrets_all[i],
            'val_regrets': val_regrets_all[i],
            'test_regrets': test_regrets_all[i],
            'train_times': train_times_all[i]
        }
        results_data['methods'].append(method_data)
    results_filename = f"{results_dir}/results.json"
    with open(results_filename, 'w') as f:
        json.dump(results_data, f, indent=4)

    # Plot results
    plot_results(results_dir, results_data)

    # Print results
    print(f"\nResults for parameters")
    for i, method_name in enumerate(methods):
        test_regrets = test_regrets_all[i]
        print(f"Test regrets {method_name} model: {test_regrets}")
    print()
    for i, method_name in enumerate(methods):
        train_times = train_times_all[i]
        print(f"Train times {method_name} model: {train_times}")

if __name__ == "__main__":
    main()
