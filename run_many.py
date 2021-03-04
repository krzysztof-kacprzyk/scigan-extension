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
    parser.add_argument("--num_treatments", default=3, type=int)
    parser.add_argument("--treatment_selection_bias", default=2.0, type=float)
    parser.add_argument("--dosage_selection_bias", default=2.0, type=float)
    parser.add_argument("--save_dataset", default=False)
    parser.add_argument("--validation_fraction", default=0.1, type=float)
    parser.add_argument("--test_fraction", default=0.2, type=float)
    parser.add_argument("--model_name", default="scigan_test")
    parser.add_argument("--verbose", action='store_true')
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
    parser.add_argument("--modify_g_loss", type=int, default=0)
    parser.add_argument("--xavier", action='store_true')
    parser.add_argument("--n", default=1, type=int)
    parser.add_argument("--source", choices=['tcga','ctg'], default='tcga')


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

    if args.source == 'tcga':
        data_class = TCGA_Data(dataset_params)
        dataset = data_class.dataset
        dataset_train, dataset_val, dataset_test = get_dataset_splits(dataset)
    elif args.source == 'ctg':
        data_class = CTG_Data(dataset_params)
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

    mise_list =[]
    dpe_list = []
    pe_list = []
    mise_dict_list = []
    dpe_dict_list = []

    gan_mise_list =[]
    gan_dpe_list = []
    gan_pe_list = []
    gan_mise_dict_list = []
    gan_dpe_dict_list = []

    for i in range(args.n):

        print("-"*20)
        print(f"Test {i+1} out of {args.n}")
        print("-"*20)
        model_name = f"{args.model_name}_{i+1}_of_{args.n}"
        export_dir = os.path.join(args.model_output,model_name)
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)

        params = {'num_features': dataset_train['x'].shape[1], 'num_treatments': args.num_treatments,
                'num_dosage_samples': args.num_dosage_samples, 'export_dir': export_dir,
                'alpha': args.alpha, 'batch_size': args.batch_size, 'h_dim': args.h_dim,
                'h_inv_eqv_dim': args.h_inv_eqv_dim, 'iterations_gan':args.iterations_gan,
                'iterations_inference':args.iterations_inference, 'agg':args.agg, 'modify_i_loss':args.modify_i_loss,
                'modify_g_loss':args.modify_g_loss, 'xavier':args.xavier}
        
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
                                            num_dosage_samples=params['num_dosage_samples'], model_folder=export_dir, use_gan=False,
                                            test_t=dataset_test['t'], test_d=dataset_test['d'], test_y=dataset_test['y_normalized'])
        
        
        
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

        if args.use_gan:

            gan_mise, gan_dpe, gan_pe, gan_mise_dict, gan_dpe_dict = compute_eval_metrics(dataset, dataset_test['x'], num_treatments=params['num_treatments'],
                                            num_dosage_samples=params['num_dosage_samples'], model_folder=export_dir, use_gan=True,
                                            test_t=dataset_test['t'], test_d=dataset_test['d'], test_y=dataset_test['y_normalized'])
        
        
            
            gan_mise_list.append(gan_mise)
            gan_dpe_list.append(gan_dpe)
            gan_pe_list.append(gan_pe)
            gan_mise_dict_list.append(gan_mise_dict)
            gan_dpe_dict_list.append(gan_dpe_dict)

            print("GAN Mise: %s" % str(gan_mise))
            print("GAN DPE: %s" % str(gan_dpe))
            print("GAN PE: %s" % str(gan_pe))

            for treatment_idx in gan_mise_dict.keys():
                print(f"Mise for treatment {treatment_idx}: {gan_mise_dict[treatment_idx]}")
            
            for treatment_idx in gan_dpe_dict.keys():
                print(f"DPE for treatment {treatment_idx}: {gan_dpe_dict[treatment_idx]}")
        

    print("-"*20)
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
    
    if args.use_gan:

        print("-"*20)
        print(f"GAN Results for all {args.n} tests")
        print("-"*20)
        print(f"GAN Average MISE: {np.mean(gan_mise_list)} | std: {np.std(gan_mise_list)}")
        print(f"GAN Average DPE: {np.mean(gan_dpe_list)} | std: {np.std(gan_dpe_list)}")
        print(f"GAN Average PE: {np.mean(gan_pe_list)} | std: {np.std(gan_pe_list)}")

        for treatment_idx in range(args.num_treatments):
            gan_mise_treatment_list = [gan_mise_dict[treatment_idx] for gan_mise_dict in gan_mise_dict_list]
            print(f"GAN Average Mise for treatment {treatment_idx}: {np.mean(gan_mise_treatment_list)} | std: {np.std(gan_mise_treatment_list)}")

        for treatment_idx in range(args.num_treatments):
            gan_dpe_treatment_list = [gan_dpe_dict[treatment_idx] for gan_dpe_dict in gan_dpe_dict_list]
            print(f"GAN Average DPE for treatment {treatment_idx}: {np.mean(gan_dpe_treatment_list)} | std: {np.std(gan_dpe_treatment_list)}")

    

