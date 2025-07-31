import tdc 
from tdc.benchmark_group import admet_group
from utils import get_rdkit_x_y, eval_TabPFN, save_json

def run_experiment():
        
    datasets = tdc.utils.retrieve_benchmark_names('ADMET_Group')
    group = admet_group(path = "data/")
    results_tdc = {}
    results_file = "results_tabpfn_test.json"

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")

        benchmark = group.get(dataset) 
        train, test = benchmark['train_val'], benchmark['test']

        print("Extracting RDKit features...")
        x_train, y_train = get_rdkit_x_y(train)
        x_test, y_test = get_rdkit_x_y(test)

        predictions_list = []
        for seed in [1, 2, 3, 4, 5]:
            print(f"Evaluating with seed: {seed}")
            y_pred_test = eval_TabPFN(x_train, y_train, x_test, seed)
            predictions_list.append({dataset: y_pred_test})

        results_dataset = group.evaluate_many(predictions_list)
        results_tdc = {**results_tdc, **results_dataset}  # Merge results into the dictionary
        print(f"Results for {dataset}: {results_dataset[dataset]}\n")
        save_json(results_tdc, results_file)

    save_json(results_tdc, results_file)


if __name__ == "__main__":
    run_experiment()