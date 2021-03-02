import argparse
import pickle
from utils.evaluation_utils import compute_eval_metrics
import os

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_treatments", default=3, type=int)
    parser.add_argument("--num_dosage_samples", default=5, type=int)
    parser.add_argument("--ds_input", default="datasets/generated")
    parser.add_argument("--models_folder", default="saved_models")
    parser.add_argument("--model_name", default="scigan_test")

    return parser.parse_args()


if __name__ == "__main__":

    args = init_arg()

    with open(os.path.join(args.ds_input,'full_dataset.pickle'), 'rb') as f:
        dataset = pickle.load(f)
    with open(os.path.join(args.ds_input,'train_dataset.pickle'), 'rb') as f:
        dataset_train = pickle.load(f)
    with open(os.path.join(args.ds_input,'val_dataset.pickle'), 'rb') as f:
        dataset_val = pickle.load(f)
    with open(os.path.join(args.ds_input,'test_dataset.pickle'), 'rb') as f:
        dataset_test = pickle.load(f)

    model_dir = os.path.join(args.models_folder,args.model_name)

    mise, dpe, pe, mise_dict, dpe_dict = compute_eval_metrics(dataset, dataset_test['x'], num_treatments=args.num_treatments,
                                         num_dosage_samples=args.num_dosage_samples, model_folder=model_dir)

    print("Mise: %s" % str(mise))
    print("DPE: %s" % str(dpe))
    print("PE: %s" % str(pe))

    for treatment_idx in mise_dict.keys():
        print(f"Mise for treatment {treatment_idx}: {mise_dict[treatment_idx]}")
    
    for treatment_idx in dpe_dict.keys():
        print(f"DPE for treatment {treatment_idx}: {dpe_dict[treatment_idx]}")

