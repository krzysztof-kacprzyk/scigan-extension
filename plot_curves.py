import numpy as np
from scipy.integrate import romb
import tensorflow as tf
import argparse
import os
import pickle

from data_simulation import get_patient_outcome

from scipy.optimize import minimize



import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def sample_dosages(batch_size, num_treatments, num_dosages):
    dosage_samples = np.random.uniform(0., 1., size=[batch_size, num_treatments, num_dosages])
    return dosage_samples


def get_model_predictions(sess, num_treatments, num_dosage_samples, test_data,
                          use_gan=False, t=None, d=None, y=None):
    batch_size = test_data['x'].shape[0]

    treatment_dosage_samples = sample_dosages(batch_size, num_treatments, num_dosage_samples)
    factual_dosage_position = np.random.randint(num_dosage_samples, size=[batch_size])
    treatment_dosage_samples[range(batch_size), test_data['t'], factual_dosage_position] = test_data['d']

    treatment_dosage_mask = np.zeros(shape=[batch_size, num_treatments, num_dosage_samples])
    treatment_dosage_mask[range(batch_size), test_data['t'], factual_dosage_position] = 1

    if use_gan:

        noise_el = np.random.uniform(0., 1.)

        noise = np.tile(noise_el, [batch_size, num_treatments * num_dosage_samples])

        one_hot_treatment = np.zeros(shape=[batch_size, num_treatments])

        one_hot_treatment[range(batch_size), np.repeat(t, batch_size)] = 1

        dosage = np.expand_dims(np.repeat(d, batch_size), axis=-1)

        y = np.expand_dims(np.repeat(y, batch_size), axis=-1)

        logits = sess.run('generator_outcomes:0',
                        feed_dict={'input_features:0': test_data['x'],
                                   'input_treatment_dosage_samples:0': treatment_dosage_samples,
                                   'input_treatment:0': one_hot_treatment,
                                   'input_dosage:0': dosage,
                                   'input_noise:0': noise,
                                   'input_y:0': y})
    else:
        logits = sess.run('inference_outcomes:0',
                            feed_dict={'input_features:0': test_data['x'],
                                    'input_treatment_dosage_samples:0': treatment_dosage_samples})

    Y_pred = np.sum(treatment_dosage_mask * logits, axis=(1, 2))

    return Y_pred


class DoseCurvePlotter:

    def __init__(self, dataset, test_patients, num_treatments, num_dosage_samples, model_folder, fig_size, fig_dpi,
                 use_gan=False, test_t=None, test_d=None, test_y=None):
        self.dataset = dataset
        self.test_patients = test_patients
        self.num_treatments = num_treatments
        self.num_dosage_samples = num_dosage_samples
        self.model_folder = model_folder
        self.fig_size = fig_size
        self.fig_dpi = fig_dpi
        self.use_gan = use_gan
        self.test_t = test_t
        self.test_d = test_d
        self.test_y = test_y


    def get_num_of_test_patients(self):
        return len(self.test_patients)

    def get_pred_dose_response_curve(self, patient_idx, treatment_idx, discrete=True, discretization_power=6):

        patient = self.test_patients[patient_idx]

        def pred_dose_response_curve(dosage):

            with tf.Session(graph=tf.Graph()) as sess:
                tf.saved_model.loader.load(sess, ["serve"], self.model_folder)

                test_data = dict()
                test_data['x'] = np.expand_dims(patient, axis=0)
                test_data['t'] = np.expand_dims(treatment_idx, axis=0)
                test_data['d'] = np.expand_dims(dosage, axis=0)

                ret_val = get_model_predictions(sess=sess, num_treatments=self.num_treatments,
                                                num_dosage_samples=self.num_dosage_samples,
                                                test_data=test_data)
                ret_val = ret_val * (self.dataset['metadata']['y_max'] - self.dataset['metadata']['y_min']) + \
                        self.dataset['metadata']['y_min']
                
                return ret_val

        if discrete:

            with tf.Session(graph=tf.Graph()) as sess:
                tf.saved_model.loader.load(sess, ["serve"], self.model_folder)
           
                num_integration_samples = 2 ** discretization_power + 1
                step_size = 1. / num_integration_samples
                treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)

                test_data = dict()
                test_data['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
                test_data['t'] = np.repeat(treatment_idx, num_integration_samples)
                test_data['d'] = treatment_strengths

                pred_dose_response = get_model_predictions(sess=sess, num_treatments=self.num_treatments,
                                                            num_dosage_samples=self.num_dosage_samples, test_data=test_data)
                pred_dose_response = pred_dose_response * (
                        self.dataset['metadata']['y_max'] - self.dataset['metadata']['y_min']) + \
                                        self.dataset['metadata']['y_min']
            
            return pred_dose_response
        
        else:
            return pred_dose_response_curve
    
    def get_pred_dose_response_curves(self, patients_treatments_list, discretization_power=6):

        pred_dose_responses = []

        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], self.model_folder)
        
            num_integration_samples = 2 ** discretization_power + 1
            step_size = 1. / num_integration_samples
            treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)

            for patient_treatment in patients_treatments_list:
                
                patient_idx = patient_treatment[0]
                treatment_idx = patient_treatment[1]

                patient = self.test_patients[patient_idx]

                test_data = dict()
                test_data['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
                test_data['t'] = np.repeat(treatment_idx, num_integration_samples)
                test_data['d'] = treatment_strengths

                t = None
                d = None
                y = None

                if self.use_gan:
                    t = self.test_t[patient_idx]
                    d = self.test_d[patient_idx]
                    y = self.test_y[patient_idx]

                pred_dose_response = get_model_predictions(sess=sess, num_treatments=self.num_treatments,
                                                            num_dosage_samples=self.num_dosage_samples, test_data=test_data, 
                                                            use_gan=self.use_gan,
                                                            t=t, d=d, y=y)
                pred_dose_response = pred_dose_response * (
                        self.dataset['metadata']['y_max'] - self.dataset['metadata']['y_min']) + \
                                        self.dataset['metadata']['y_min']
                
                pred_dose_responses.append(pred_dose_response)
        
        return pred_dose_responses
        

    
    def get_true_dose_response_curve(self, patient_idx, treatment_idx, discrete=True, discretization_power=6):
        
        patient = self.test_patients[patient_idx]

        def true_dose_response_curve(dosage):
            y = get_patient_outcome(patient, self.dataset['metadata']['v'], treatment_idx, dosage)
            return y

        if discrete:
            num_integration_samples = 2 ** discretization_power + 1
            step_size = 1. / num_integration_samples
            treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)

            return [true_dose_response_curve(d) for d in treatment_strengths]
        
        else:
            return true_dose_response_curve

    def plot_dose_curve(self, patient_idx, treatment_idx, discretization_power=6):

        true_dose_response = self.get_true_dose_response_curve(patient_idx, treatment_idx, discrete=True,
                                                                discretization_power=discretization_power)
        pred_dose_response = self.get_pred_dose_response_curve(patient_idx, treatment_idx, discrete=True,
                                                                discretization_power=discretization_power)                                        

        num_integration_samples = 2 ** discretization_power + 1
        step_size = 1. / num_integration_samples
        treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)

        plt.plot(treatment_strengths, true_dose_response, label='True')
        plt.plot(treatment_strengths, pred_dose_response, label='Predicted')

        plt.legend()

        plt.show()

    def plot_random_dose_curves(self, nrows, ncols, discretization_power=6):

        num_integration_samples = 2 ** discretization_power + 1
        step_size = 1. / num_integration_samples
        treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)

        fig, axs = plt.subplots(nrows, ncols, figsize=self.fig_size, dpi=self.fig_dpi)

        for row in range(nrows):
            for col in range(ncols):

                patient_idx = np.random.randint(self.get_num_of_test_patients())
                treatment_idx = np.random.randint(self.num_treatments)

                true_dose_response = self.get_true_dose_response_curve(patient_idx, treatment_idx, discrete=True,
                                                                discretization_power=discretization_power)
                pred_dose_response = self.get_pred_dose_response_curve(patient_idx, treatment_idx, discrete=True,
                                                                discretization_power=discretization_power)                                        

                axs[row, col].plot(treatment_strengths, true_dose_response, label='True')
                axs[row, col].plot(treatment_strengths, pred_dose_response, label='Predicted')
                axs[row, col].set_title(f"Treatment: {treatment_idx}")

        plt.tight_layout()
        plt.show()

    def plot_sample_dose_curves(self, discretization_power=6):

        num_integration_samples = 2 ** discretization_power + 1
        step_size = 1. / num_integration_samples
        treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)

        nrows = self.num_treatments
        ncols = 6

        tenth_num_patients = self.get_num_of_test_patients() // 10

        patient_idx_list = [i * tenth_num_patients for i in range(ncols)]

        patients_treatments_list = list(zip(patient_idx_list * 3, [i//ncols for i in range(ncols * nrows)]))

        pred_dose_responses = self.get_pred_dose_response_curves(patients_treatments_list)

        fig, axs = plt.subplots(nrows, ncols, figsize=self.fig_size, dpi=self.fig_dpi)

        for row in range(nrows):
            for col in range(ncols):

                pred_dose_response = pred_dose_responses[row * ncols + col]

                true_dose_response = self.get_true_dose_response_curve(patient_idx_list[col], row, discrete=True,
                                                                discretization_power=discretization_power)                              

                axs[row, col].plot(treatment_strengths, true_dose_response, label='True')
                axs[row, col].plot(treatment_strengths, pred_dose_response, label='Predicted')
                axs[row, col].set_title(f"Treatment: {row}")
       
        plt.tight_layout()
        plt.show()

    

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_treatments", default=3, type=int)
    parser.add_argument("--num_dosage_samples", default=5, type=int)
    parser.add_argument("--ds_input", default="datasets/generated")
    parser.add_argument("--models_folder", default="saved_models")
    parser.add_argument("--model_name", default="scigan_test")
    parser.add_argument("--fig_size_h", default=4, type=int)
    parser.add_argument("--fig_size_w", default=6, type=int)
    parser.add_argument("--fig_dpi", default=100, type=int)
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


    plotter = DoseCurvePlotter(dataset, dataset_test['x'], num_treatments=args.num_treatments,
                                         num_dosage_samples=args.num_dosage_samples, model_folder=model_dir,
                                         fig_size=(args.fig_size_w, args.fig_size_h), fig_dpi=args.fig_dpi,
                                         use_gan=args.use_gan,
                                         test_t=chosen_ds['t'], test_d=chosen_ds['d'], test_y=chosen_ds['y_normalized'])
    
    plotter.plot_sample_dose_curves()



