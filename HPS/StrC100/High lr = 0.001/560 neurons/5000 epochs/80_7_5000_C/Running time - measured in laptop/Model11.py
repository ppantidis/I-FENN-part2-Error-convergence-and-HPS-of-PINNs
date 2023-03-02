import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from mpl_toolkits.mplot3d import Axes3D
import time
import pickle
import gc
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import StandardScaler

###########################################################################################################################################
# Define hyperparameters 
# np.random.seed(1234)
# tf.set_random_seed(1234)        
scaler = StandardScaler()
layers = [4, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1]
epochs = 5000
hist_vec = []
num_hidlayers = len(layers) - 2
num_hidlayer_neurons = layers[1]
IsExistModel = 0
learning_rate = 0.001
gc.enable()
Modelmesh = "StrC100"

###########################################################################################################################################
class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X, X_lrb, X_btb, layers, ExistModel=IsExistModel, modelDir=None):
        
        # -----------------------------------------------------------------------------------------------------------------------
        # Access each column (xcoord, ycoord, elocal and g) of the dataset for predictions at collocation points
        self.xcoord         = X[:,0:1]
        self.ycoord         = X[:,1:2]
        self.g              = X[:,2:3]
        self.elocal         = X[:,3:4]

        # Access each column (xcoord, ycoord, elocal and g) of the dataset for predictions derivatives at left/right boundary points
        self.xcoord_lrb         = X_lrb[:,0:1]
        self.ycoord_lrb         = X_lrb[:,1:2]
        self.g_lrb              = X_lrb[:,2:3]
        self.elocal_lrb         = X_lrb[:,3:4]

        # Access each column (xcoord, ycoord, elocal and g) of the dataset for predictions derivatives at bottom/top boundary points
        self.xcoord_btb         = X_btb[:,0:1]
        self.ycoord_btb         = X_btb[:,1:2]
        self.g_btb              = X_btb[:,2:3]
        self.elocal_btb         = X_btb[:,3:4]

        # -----------------------------------------------------------------------------------------------------------------------
        # Define layers        
        self.layers = layers

        # Initialize NNs
        if ExistModel == 0:
            self.weights, self.biases = self.initialize_NN(self.layers)
        elif ExistModel == 1:
            self.weights, self.biases = self.load_NN(modelDir, self.layers)
        else:
            print("Check your IsExistModel variable")

        # -----------------------------------------------------------------------------------------------------------------------
        # TF.Placeholders
        # Placeholders for predictions at collocation points 
        self.xcoord_tf         = tf.placeholder(tf.float32, shape=[None, self.xcoord.shape[1]])
        self.ycoord_tf         = tf.placeholder(tf.float32, shape=[None, self.ycoord.shape[1]])
        self.g_tf              = tf.placeholder(tf.float32, shape=[None, self.g.shape[1]])        
        self.elocal_tf         = tf.placeholder(tf.float32, shape=[None, self.elocal.shape[1]])
        
        # Placeholders for prediction derivatives at left/right boundary points          
        self.xcoord_lrb_tf         = tf.placeholder(tf.float32, shape=[None, self.xcoord_lrb.shape[1]])
        self.ycoord_lrb_tf         = tf.placeholder(tf.float32, shape=[None, self.ycoord_lrb.shape[1]])
        self.g_lrb_tf              = tf.placeholder(tf.float32, shape=[None, self.g_lrb.shape[1]])        
        self.elocal_lrb_tf         = tf.placeholder(tf.float32, shape=[None, self.elocal_lrb.shape[1]])
   
        # Placeholders for prediction derivatives at bottom/top boundary points          
        self.xcoord_btb_tf         = tf.placeholder(tf.float32, shape=[None, self.xcoord_btb.shape[1]])
        self.ycoord_btb_tf         = tf.placeholder(tf.float32, shape=[None, self.ycoord_btb.shape[1]])
        self.g_btb_tf              = tf.placeholder(tf.float32, shape=[None, self.g_btb.shape[1]])        
        self.elocal_btb_tf         = tf.placeholder(tf.float32, shape=[None, self.elocal_btb.shape[1]])
   

        # -----------------------------------------------------------------------------------------------------------------------
        # tf Graphs
        _ , _ , self.enonlocal_pred_x_lrb, _                    = self.net_pred(self.xcoord_lrb_tf, self.ycoord_lrb_tf, self.g_lrb_tf, self.elocal_lrb_tf)  # Solution derivative prediction at left/right boundary points
        _ , _ , _ , self.enonlocal_pred_y_btb                   = self.net_pred(self.xcoord_btb_tf, self.ycoord_btb_tf, self.g_btb_tf, self.elocal_btb_tf)  # Solution derivative prediction at bottom/top boundary points
        self.enonlocal_pred, self.enonlocal_pred_elocal, _ , _  = self.net_pred(self.xcoord_tf, self.ycoord_tf, self.g_tf, self.elocal_tf)                  # Solution prediction at collocation points
        self.pde_pred                                           = self.net_pde_pred(self.xcoord_tf, self.ycoord_tf, self.g_tf, self.elocal_tf)              # Residual prediction at collocation points
        

        # -----------------------------------------------------------------------------------------------------------------------
        if ExistModel == 0:
            # Loss
            self.loss = tf.norm(self.enonlocal_pred_x_lrb, ord="euclidean") + \
                        tf.norm(self.enonlocal_pred_y_btb, ord="euclidean") + \
                        tf.norm(self.pde_pred, ord="euclidean")             
            
            # Optimizers
            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                    method = 'L-BFGS-B', 
                                                                    options = {'maxiter': 50000,
                                                                            'maxfun': 50000,
                                                                            'maxcor': 50,
                                                                            'maxls': 50,
                                                                            'ftol' : 0.000001 * np.finfo(float).eps})
        
            self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = learning_rate)
            self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                    
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                        log_device_placement=True))

        
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Weights logger 
        self.weights_log = []
        self.biases_log = []
        
    # ---------------------------------------------------------------------------------------------------------------------------------------              
    def initialize_NN(self, layers):        
        
        weights = []
        biases  = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        
        return weights, biases 

    # ---------------------------------------------------------------------------------------------------------------------------------------        
    def xavier_init(self, size):
        
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    # ---------------------------------------------------------------------------------------------------------------------------------------
    def save_NN(self, fileDir):
        weights = self.sess.run(self.weights)
        biases = self.sess.run(self.biases)
        with open(fileDir, 'wb') as f:
            pickle.dump([weights, biases], f)
            print("Saved NN parameters successfully!")
                

    # ---------------------------------------------------------------------------------------------------------------------------------------
    def load_NN(self, fileDir, layers):
        weights = []
        biases = []
        with open(fileDir, 'rb') as f:
            weights, biases = pickle.load(f)
        return weights, biases

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # neural_net: executes forward propagation -> returns prediction
    def neural_net(self, X, weights, biases):
        
        num_layers = len(weights) + 1
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            X = tf.tanh(tf.add(tf.matmul(X, W), b))
        W = weights[-1]
        b = biases[-1]
        y_pred = tf.add(tf.matmul(X, W), b)
        
        return y_pred
    
    # ---------------------------------------------------------------------------------------------------------------------------------------
    # net_pred: calls forward propagation and tf.gradients to return a) prediction and b) first-order derivatives wrt x, y coordinates and local strain
    def net_pred(self, xcoord, ycoord, g, elocal):
        
        X = tf.concat([xcoord,ycoord,g,elocal],1)
        enonlocal_pred          = self.neural_net(X, self.weights, self.biases) # Prediction of nonlocal strain 
        enonlocal_pred_x        = tf.gradients(enonlocal_pred, xcoord)[0]       # First-order derivative of nonlocal strain prediction w.r.t. the x-coordinate
        enonlocal_pred_y        = tf.gradients(enonlocal_pred, ycoord)[0]       # First-order derivative of nonlocal strain prediction w.r.t. the y-coordinate
        enonlocal_pred_elocal   = tf.gradients(enonlocal_pred, elocal)[0]       # First-order derivative of nonlocal strain prediction w.r.t. the local equivalent strain

        return enonlocal_pred, enonlocal_pred_elocal, enonlocal_pred_x, enonlocal_pred_y

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # net_pde_pred: calls net_pred and calculates second-order derivatives wrt x and y coordinates -> returns pde residual
    def net_pde_pred(self, xcoord, ycoord, g, elocal):

        enonlocal_pred, _, enonlocal_pred_x, enonlocal_pred_y = self.net_pred(xcoord, ycoord, g, elocal)
        enonlocal_pred_xx = tf.gradients(enonlocal_pred_x, xcoord)[0]
        enonlocal_pred_yy = tf.gradients(enonlocal_pred_y, ycoord)[0]
        pde_pred = enonlocal_pred - g * (enonlocal_pred_xx + enonlocal_pred_yy) - elocal
        
        return pde_pred
    
    # ---------------------------------------------------------------------------------------------------------------------------------------
    def callback(self, loss):
        hist_vec.append(loss)
        print('Loss:', loss)

    # ---------------------------------------------------------------------------------------------------------------------------------------        
    def train(self, nIter):
        
        tf_dict = { # Predictions at collocation points
                    self.xcoord_tf             : self.xcoord, 
                    self.ycoord_tf             : self.ycoord,
                    self.g_tf                  : self.g, 
                    self.elocal_tf             : self.elocal,
                    # Prediction derivatives at left/right boundary points
                    self.xcoord_lrb_tf         : self.xcoord_lrb, 
                    self.ycoord_lrb_tf         : self.ycoord_lrb,
                    self.g_lrb_tf              : self.g_lrb, 
                    self.elocal_lrb_tf         : self.elocal_lrb,
                    # Prediction derivatives at bottom/top boundary points
                    self.xcoord_btb_tf         : self.xcoord_btb, 
                    self.ycoord_btb_tf         : self.ycoord_btb,
                    self.g_btb_tf              : self.g_btb, 
                    self.elocal_btb_tf         : self.elocal_btb}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            elapsed = time.time() - start_time
            loss_value = self.sess.run(self.loss, tf_dict)
            hist_vec.append(loss_value)
            # Print
            if (it + 1) % 100 == 0:
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it + 1, loss_value, elapsed))
                start_time = time.time()

                print("Weights stored...")
                weights = self.sess.run(self.weights)
                biases = self.sess.run(self.biases)
                self.weights_log.append(weights)
                self.biases_log.append(biases)

            if it + 1 == epochs:
                enonlocal_pred_star, _, _ = model.predict(xtraindata, xtraindata_lrb, xtraindata_btb)
                print("it + 1: ", it + 1)
                np.savetxt("Model11_TestE_" + Modelmesh + "_Adam_Predictions_" + str(num_hidlayers) + "hidlay_" + str(num_hidlayer_neurons) + "units_" + str(epochs) + "ep_inc_164" + "_v"+ str(num_cases + 1) + ".txt", enonlocal_pred_star, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)


        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)


    # ---------------------------------------------------------------------------------------------------------------------------------------                                    
    def predict(self, X_star, X_lrb_star, X_btb_star):
        
        tf_dict = { # Predictions at collocation points
                    self.xcoord_tf:             X_star[:,0:1], 
                    self.ycoord_tf:             X_star[:,1:2], 
                    self.g_tf:                  X_star[:,2:3], 
                    self.elocal_tf:             X_star[:,3:4],
                    # Prediction derivatives at left/right boundary points
                    self.xcoord_lrb_tf:         X_lrb_star[:,0:1], 
                    self.ycoord_lrb_tf:         X_lrb_star[:,1:2], 
                    self.g_lrb_tf:              X_lrb_star[:,2:3], 
                    self.elocal_lrb_tf:         X_lrb_star[:,3:4],
                    # Prediction derivatives at bottom/top boundary points
                    self.xcoord_btb_tf:         X_btb_star[:,0:1], 
                    self.ycoord_btb_tf:         X_btb_star[:,1:2], 
                    self.g_btb_tf:              X_btb_star[:,2:3], 
                    self.elocal_btb_tf:         X_btb_star[:,3:4]}

        enonlocal_pred_star         = self.sess.run(self.enonlocal_pred, tf_dict)
        enonlocal_pred_elocal_star  = self.sess.run(self.enonlocal_pred_elocal, tf_dict)
        pde_star                    = self.sess.run(self.pde_pred, tf_dict)
               
        return enonlocal_pred_star, enonlocal_pred_elocal_star, pde_star


###########################################################################################################################################
def compute_weights_diff(weights_1, weights_2):
    weights = []
    N = len(weights_1)
    for k in range(N):
        weight = weights_1[k] - weights_2[k]
        weights.append(weight)
        # print(weight)
    return weights

def compute_weights_norm(weights, biases):
    norm = 0
    for w in weights:
        norm = norm + np.sum(np.square(w))
    for b in biases:
        norm = norm + np.sum(np.square(b))
    norm = np.sqrt(norm)
    return norm


###########################################################################################################################################
if __name__ == "__main__": 
    
    with tf.device('/device:GPU:0'):

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Enable GPU -> still in progress 
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Load, preprocess and print the traindata
        column_list = ["xcoord", "ycoord", "g", "elocal"]
        data     = pd.read_csv("data_GP_" + Modelmesh + "_Nonlocal_gradient_Analytical_inc_164.csv", usecols = column_list)         # pandas dataframe
        data_lrb = pd.read_csv("data_nodes_lrb_" + Modelmesh + "_Nonlocal_gradient_Analytical_inc_164.csv", usecols = column_list)  # pandas dataframe
        data_btb = pd.read_csv("data_nodes_btb_" + Modelmesh + "_Nonlocal_gradient_Analytical_inc_164.csv", usecols = column_list)  # pandas dataframe
        xtraindata = np.column_stack((data.xcoord.values,data.ycoord.values,data.g.values,data.elocal.values))                         # Numpy array of Size: # GPs x 5
        xtraindata_lrb = np.column_stack((data_lrb.xcoord.values,data_lrb.ycoord.values,data_lrb.g.values,data_lrb.elocal.values)) # Numpy array of Size: # left/right boundary x 5
        xtraindata_btb = np.column_stack((data_btb.xcoord.values,data_btb.ycoord.values,data_btb.g.values,data_btb.elocal.values)) # Numpy array of Size: # bottom/top boundary nodes x 5

        with pd.option_context('display.precision', 20):
            print("data: ", data)
            print("data_lrb: ", data_lrb)
            print("data_btb: ", data_btb)
        
        for num_cases in range(0, 10):
            
            hist_vec = []

            print("current case number: ", num_cases + 1)
            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Generate the model and produce results
            if IsExistModel == 0:
                
                # Option 1: train from scratch
                model       = PhysicsInformedNN(xtraindata, xtraindata_lrb, xtraindata_btb, layers)
                start_time  = time.time()                
                model.train(epochs)
                elapsed = time.time() - start_time                
                print('Training time: %.4f' % (elapsed))
                model.save_NN("Trained_model_elastic_" + str(num_hidlayers) + "hidlay_" + str(num_hidlayer_neurons) + "units_" + str(epochs) + "ep" + "_v"+ str(num_cases + 1) + ".pickle")
              
               # Testcase: use the just trained network to make predictions and store results
                predictdata            = pd.read_csv("data_GP_" + Modelmesh + "_Nonlocal_gradient_Analytical_inc_164.csv", usecols = column_list)           # pandas dataframe
                predictdata_lrb        = pd.read_csv("data_nodes_lrb_" + Modelmesh + "_Nonlocal_gradient_Analytical_inc_164.csv", usecols = column_list)    # pandas dataframe
                predictdata_btb        = pd.read_csv("data_nodes_btb_" + Modelmesh + "_Nonlocal_gradient_Analytical_inc_164.csv", usecols = column_list)    # pandas dataframe

                predictxtraindata      = np.column_stack((predictdata.xcoord.values,predictdata.ycoord.values,predictdata.g.values,predictdata.elocal.values))                        # Numpy array of Size: # GPs x 5
                predictxtraindata_lrb  = np.column_stack((predictdata_lrb.xcoord.values,predictdata_lrb.ycoord.values,predictdata_lrb.g.values,predictdata_lrb.elocal.values))    # Numpy array of Size: # left/right boundary x 5
                predictxtraindata_btb  = np.column_stack((predictdata_btb.xcoord.values,predictdata_btb.ycoord.values,predictdata_btb.g.values,predictdata_btb.elocal.values))    # Numpy array of Size: # bottom/top boundary nodes x 5

                enonlocal_pred_star, enonlocal_pred_elocal_star, pde_star = model.predict(predictxtraindata, predictxtraindata_lrb, predictxtraindata_btb)

                # Restore the list weights and biases
                weights_log = model.weights_log
                biases_log = model.biases_log

                # Norm of the weights at initialization
                weights_change_list = []
                
                N = len(weights_log)
                for k in range(N):
                    weights_diff            = compute_weights_diff(weights_log[k], weights_log[k-1])
                    biases_diff             = compute_weights_diff(biases_log[k], biases_log[k-1])
                    weights_diff_norm       = compute_weights_norm(weights_diff, biases_diff)
                    weights_change          = weights_diff_norm / compute_weights_norm(weights_log[k-1], biases_log[k-1])
                    weights_change_list.append(weights_change)
                    
                np.savetxt("Model11_TestE_" + Modelmesh + "_Predictions_"               + str(num_hidlayers) + "hidlay_" + str(num_hidlayer_neurons) + "units_" + str(epochs) + "ep_inc_164" + "_v"+ str(num_cases + 1) + ".txt", enonlocal_pred_star, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
                np.savetxt("Model11_TestE_" + Modelmesh + "_Predictions_derivatives_"   + str(num_hidlayers) + "hidlay_" + str(num_hidlayer_neurons) + "units_" + str(epochs) + "ep_inc_164" + "_v"+ str(num_cases + 1) + ".txt", enonlocal_pred_elocal_star, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
                np.savetxt("Model11_TestE_" + Modelmesh + "_History_"                   + str(num_hidlayers) + "hidlay_" + str(num_hidlayer_neurons) + "units_" + str(epochs) + "ep_inc_164" + "_v"+ str(num_cases + 1) + ".txt", hist_vec, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
                np.savetxt("Model11_TestE_" + Modelmesh + "_Weights_Change_"            + str(num_hidlayers) + "hidlay_" + str(num_hidlayer_neurons) + "units_" + str(epochs) + "ep_inc_164" + "_v"+ str(num_cases + 1) + ".txt", weights_change_list, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
                
                gc.collect()

            elif IsExistModel == 1:

                # Option 2: load existing model and make predictions on the testcase
                testpredictdata             = pd.read_csv("data_GP_" + Modelmesh + "_Nonlocal_gradient_Analytical_inc_164.csv", usecols = column_list)         # pandas dataframe
                testpredictdata_lrb         = pd.read_csv("data_nodes_lrb_" + Modelmesh + "_Nonlocal_gradient_Analytical_inc_164.csv", usecols = column_list)  # pandas dataframe       
                testpredictdata_btb         = pd.read_csv("data_nodes_btb_" + Modelmesh + "_Nonlocal_gradient_Analytical_inc_164.csv", usecols = column_list)  # pandas dataframe       
                    
                testpredictxtraindata       = np.column_stack((testpredictdata.xcoord.values,testpredictdata.ycoord.values,testpredictdata.g.values,testpredictdata.elocal.values))                       # Numpy array of Size: # GPs x 5
                testpredictxtraindata_lrb   = np.column_stack((testpredictdata_lrb.xcoord.values,testpredictdata_lrb.ycoord.values,testpredictdata_lrb.g.values,testpredictdata_lrb.elocal.values))   # Numpy array of Size: # left/right boundary x 5
                testpredictxtraindata_btb   = np.column_stack((testpredictdata_btb.xcoord.values,testpredictdata_btb.ycoord.values,testpredictdata_btb.g.values,testpredictdata_btb.elocal.values))   # Numpy array of Size: # bottom/top boundary nodes x 5

                model = PhysicsInformedNN(testpredictxtraindata, testpredictxtraindata_lrb, testpredictxtraindata_btb, layers, modelDir = "Trained_model_elastic_" + str(num_hidlayers) + "hidlay_" + str(num_hidlayer_neurons) + "units_" + str(epochs) + "ep"+ ".pickle")

                testenonlocal_pred_star, testenonlocal_pred_elocal_star, testpde_star = model.predict(testpredictxtraindata,testpredictxtraindata_lrb,testpredictxtraindata_btb)

                np.savetxt("AA_Model11_TestE_" + Modelmesh + "_Predictions_"             + str(num_hidlayers) + "hidlay_" + str(num_hidlayer_neurons) + "units_" + str(epochs) + "ep" + ".txt", testenonlocal_pred_star, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
                np.savetxt("AA_Model11_TestE_" + Modelmesh + "_Predictions_derivatives_" + str(num_hidlayers) + "hidlay_" + str(num_hidlayer_neurons) + "units_" + str(epochs) + "ep" + ".txt", testenonlocal_pred_elocal_star, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)

            else:

                print("Check your IsExistModel variable!") 

        
    
    
    


