# Copyright (c) 2020, Ioana Bica

import argparse
import os
import shutil
import tensorflow as tf
import pickle

from data_simulation import get_dataset_splits, TCGA_Data
from SCIGAN import SCIGAN_Model
from SCIGAN_deep import SCIGAN_deep_Model, SCIGAN_deep_2_Model, SCIGAN_deep_3_Model, SCIGAN_deep_4_Model
from utils.evaluation_utils import compute_eval_metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_treatments", default=3, type=int)
    parser.add_argument("--treatment_selection_bias", default=2.0, type=float)
    parser.add_argument("--dosage_selection_bias", default=2.0, type=float)
    parser.add_argument("--save_dataset", default=False)
    parser.add_argument("--validation_fraction", default=0.1, type=float)
    parser.add_argument("--test_fraction", default=0.2, type=float)
    parser.add_argument("--model_name", default="scigan_test")
    parser.add_argument("--verbose", default=False)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--h_dim", default=64, type=int)
    parser.add_argument("--h_inv_eqv_dim", default=64, type=int)
    parser.add_argument("--num_dosage_samples", default=5, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--filepath", default="datasets/tcga.p")
    parser.add_argument("--ds_output", default="datasets/generated")
    parser.add_argument("--model_output", default="saved_models")
    parser.add_argument("--iterations_gan", default=5000, type=int)
    parser.add_argument("--iterations_inference", default=10000, type=int)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--v", default=1, type=int)
    parser.add_argument("--agg", choices=['sum', 'l1', 'l2', 'inf'], default='sum')
    parser.add_argument("--use_gan", action='store_true')
    parser.add_argument("--modify_i_loss", type=int, default=0)


    return parser.parse_args()


if __name__ == "__main__":

    args = init_arg()

    dataset_params = dict()
    dataset_params['num_treatments'] = args.num_treatments
    dataset_params['treatment_selection_bias'] = args.treatment_selection_bias
    dataset_params['dosage_selection_bias'] = args.dosage_selection_bias
    dataset_params['save_dataset'] = args.save_dataset
    dataset_params['validation_fraction'] = args.validation_fraction
    dataset_params['test_fraction'] = args.test_fraction
    dataset_params['filepath'] = args.filepath

    data_class = TCGA_Data(dataset_params)
    dataset = data_class.dataset
    dataset_train, dataset_val, dataset_test = get_dataset_splits(dataset)

    if args.save_dataset:

        with open(os.path.join(args.ds_output,'full_dataset.pickle'), 'wb') as f:
            pickle.dump(dataset, f)
        
        with open(os.path.join(args.ds_output,'train_dataset.pickle'), 'wb') as f:
            pickle.dump(dataset_train, f)
        
        with open(os.path.join(args.ds_output,'val_dataset.pickle'), 'wb') as f:
            pickle.dump(dataset_val, f)

        with open(os.path.join(args.ds_output,'test_dataset.pickle'), 'wb') as f:
            pickle.dump(dataset_test, f)


    export_dir = os.path.join(args.model_output,args.model_name)
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    params = {'num_features': dataset_train['x'].shape[1], 'num_treatments': args.num_treatments,
              'num_dosage_samples': args.num_dosage_samples, 'export_dir': export_dir,
              'alpha': args.alpha, 'batch_size': args.batch_size, 'h_dim': args.h_dim,
              'h_inv_eqv_dim': args.h_inv_eqv_dim, 'iterations_gan':args.iterations_gan,
              'iterations_inference':args.iterations_inference, 'agg':args.agg, 'modify_i_loss':args.modify_i_loss}
    if args.deep:
        if args.v == 1:
            model_baseline = SCIGAN_deep_Model(params)
        elif args.v == 2:
            model_baseline = SCIGAN_deep_2_Model(params)
        elif args.v == 3:
            model_baseline = SCIGAN_deep_3_Model(params)
        elif args.v == 4:
            model_baseline = SCIGAN_deep_4_Model(params)

    else:
        model_baseline = SCIGAN_Model(params)

    model_baseline.train(Train_X=dataset_train['x'], Train_T=dataset_train['t'], Train_D=dataset_train['d'],
                         Train_Y=dataset_train['y_normalized'], verbose=args.verbose)

    mise, dpe, pe, mise_dict, dpe_dict = compute_eval_metrics(dataset, dataset_test['x'], num_treatments=params['num_treatments'],
                                         num_dosage_samples=params['num_dosage_samples'], model_folder=export_dir, use_gan=args.use_gan,
                                         test_t=dataset_test['t'], test_d=dataset_test['d'], test_y=dataset_test['y_normalized'])

    print("Mise: %s" % str(mise))
    print("DPE: %s" % str(dpe))
    print("PE: %s" % str(pe))

    for treatment_idx in mise_dict.keys():
        print(f"Mise for treatment {treatment_idx}: {mise_dict[treatment_idx]}")
    
    for treatment_idx in dpe_dict.keys():
        print(f"DPE for treatment {treatment_idx}: {dpe_dict[treatment_idx]}")
