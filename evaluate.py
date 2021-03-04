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
    parser.add_argument("--use_gan", action='store_true')
    parser.add_argument("--ds_type", choices=['train','val','test'], default='test')

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

    if args.ds_type == 'train':
        chosen_ds = dataset_train
    elif args.ds_type == 'val':
        chosen_ds = dataset_val
    else:
        chosen_ds = dataset_test

    mise, dpe, pe, mise_dict, dpe_dict = compute_eval_metrics(dataset, chosen_ds['x'], num_treatments=args.num_treatments,
                                         num_dosage_samples=args.num_dosage_samples, model_folder=model_dir, use_gan=args.use_gan,
                                         test_t=chosen_ds['t'], test_d=chosen_ds['d'], test_y=chosen_ds['y_normalized'])

    print("Mise: %s" % str(mise))
    print("DPE: %s" % str(dpe))
    print("PE: %s" % str(pe))

    for treatment_idx in mise_dict.keys():
        print(f"Mise for treatment {treatment_idx}: {mise_dict[treatment_idx]}")
    
    for treatment_idx in dpe_dict.keys():
        print(f"DPE for treatment {treatment_idx}: {dpe_dict[treatment_idx]}")

