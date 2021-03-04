import argparse
import os
import shutil
import tensorflow as tf
import pickle
import numpy as np

from data_simulation import get_dataset_splits, TCGA_Data, CTG_Data
from SCIGAN import SCIGAN_Model
from SCIGAN_deep import SCIGAN_deep_Model, SCIGAN_deep_2_Model, SCIGAN_deep_3_Model, SCIGAN_deep_4_Model
from utils.evaluation_utils import compute_eval_metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def init_arg():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_treatments", default=3, type=int)
    parser.add_argument("--num_dosage_samples", default=5, type=int)
    parser.add_argument("--ds_input", default="datasets/generated")
    parser.add_argument("--models_folder", default="saved_models")
    parser.add_argument("--model_name", default="scigan_test")
    parser.add_argument("--use_gan", action='store_true')
    parser.add_argument("--ds_type", choices=['train','val','test'], default='test')
    parser.add_argument("--n", type=int, default=1)


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

    if args.ds_type == 'train':
        chosen_ds = dataset_train
    elif args.ds_type == 'val':
        chosen_ds = dataset_val
    else:
        chosen_ds = dataset_test


    mise_list =[]
    dpe_list = []
    pe_list = []
    mise_dict_list = []
    dpe_dict_list = []

    for i in range(args.n):

        print("-"*20)
        print(f"Evaluate {i+1} out of {args.n}")
        print("-"*20)
        model_name = f"{args.model_name}_{i+1}_of_{args.n}"
        export_dir = os.path.join(args.models_folder,model_name)
        

        mise, dpe, pe, mise_dict, dpe_dict = compute_eval_metrics(dataset, chosen_ds['x'], num_treatments=args.num_treatments,
                                         num_dosage_samples=args.num_dosage_samples, model_folder=export_dir, use_gan=args.use_gan,
                                         test_t=chosen_ds['t'], test_d=chosen_ds['d'], test_y=chosen_ds['y_normalized'])
        
        
        
        mise_list.append(mise)
        dpe_list.append(dpe)
        pe_list.append(pe)
        mise_dict_list.append(mise_dict)
        dpe_dict_list.append(dpe_dict)

        print("Mise: %s" % str(mise))
        print("DPE: %s" % str(dpe))
        print("PE: %s" % str(pe))

        for treatment_idx in mise_dict.keys():
            print(f"Mise for treatment {treatment_idx}: {mise_dict[treatment_idx]}")
        
        for treatment_idx in dpe_dict.keys():
            print(f"DPE for treatment {treatment_idx}: {dpe_dict[treatment_idx]}")
        

    print("-"*20)
    if args.use_gan:
        print(f"GAN Results for all {args.n} tests")
    else:
        print(f"Results for all {args.n} tests")
    print("-"*20)
    print(f"Average MISE: {np.mean(mise_list)} | std: {np.std(mise_list)}")
    print(f"Average DPE: {np.mean(dpe_list)} | std: {np.std(dpe_list)}")
    print(f"Average PE: {np.mean(pe_list)} | std: {np.std(pe_list)}")

    for treatment_idx in range(args.num_treatments):
        mise_treatment_list = [mise_dict[treatment_idx] for mise_dict in mise_dict_list]
        print(f"Average Mise for treatment {treatment_idx}: {np.mean(mise_treatment_list)} | std: {np.std(mise_treatment_list)}")

    for treatment_idx in range(args.num_treatments):
        dpe_treatment_list = [dpe_dict[treatment_idx] for dpe_dict in dpe_dict_list]
        print(f"Average DPE for treatment {treatment_idx}: {np.mean(dpe_treatment_list)} | std: {np.std(dpe_treatment_list)}")
    

    

