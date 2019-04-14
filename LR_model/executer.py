import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import general_func as gf
import binning
import feature_inspection
import model_inspection
from importlib import reload
binning = reload(binning)
model_inspection = reload(model_inspection)
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.linear_model import LogisticRegression
import pdb
#    pdb.set_trace()

class run:
    def __init__(self, all_data, ins, oot, columnX, columnY):
        self.fi = feature_inspection.feature_inspection()
        self.bi = binning.binning()
        self.mi = model_inspection.model_inspection()
        self.all_data = all_data
        self.ins = ins
        self.oot = oot
        self.columnX = columnX
        self.columnY = columnY
        
    def assign_para(self):
        self.nan_th = 0.6
        self.zero_th = 0.9
        self.auc_th = 0.51
        self.bin_initial_number = 5
        self.min_split_sample = 1000
        self.bin_method = 'chi2'
        self.if_mono = True
        self.stop_condition = 3
        self.psi_th = 0.03
        self.stability_th = 0.5
        self.tsi_th = 0.01
        self.iv_ins_th = 0.005
        self.iv_oot_th = 0.005
        self.lasso_list = [.00001]
    
    def basic_inspection(self):
        self.basic_statistic = self.fi.all_feature_inspection(self.all_data, self.columnX, self.columnY) 
        self.columnX_1st = self.basic_statistic[(self.basic_statistic['nan_ratio']<self.nan_th)
                                                  &(self.basic_statistic['zero_rate']<self.zero_th)
                                                  &(self.basic_statistic['auc']>self.auc_th)].index
        
    def advanced_inspection(self):
        train_features = self.ins[self.columnX_1st]
        train_label = self.ins[self.columnY]
        test_features = self.oot[self.columnX_1st]
        test_label = self.oot[self.columnY]
        self.advanced_statistic = self.bi.all_feature_psi(train_features, train_label,
                                test_features, test_label, self.basic_statistic,
                                self.bin_initial_number, self.min_split_sample, self.bin_method,
                                self.if_mono, self.stop_condition)
        self.columnX_2nd = self.advanced_statistic[(self.advanced_statistic['psi']<self.psi_th)
                                        &(abs(self.advanced_statistic['iv_train']-self.advanced_statistic['iv_test'])/(self.advanced_statistic['iv_train']+self.advanced_statistic['iv_train'])<self.stability_th)
                                        &(self.advanced_statistic['iv_train']>self.iv_ins_th)
                                        &(self.advanced_statistic['iv_test']>self.iv_oot_th)].index
        
    def features_to_bins(self):
        self.train_woe, _, self.feature_dict = self.bi.all_features_to_bins(self.ins, self.columnX_2nd, self.columnY, self.advanced_statistic)
        self.test_woe, _, _ = self.bi.all_features_to_bins(self.oot, self.columnX_2nd, self.columnY, self.advanced_statistic, self.feature_dict['bin_woe'])
        
    def lasso_selection(self):
        train_woe_2ndX = self.train_woe[self.columnX_2nd]
        trainY = self.ins[self.columnY].values.flatten()
#        test_woe_2ndX = self.test_woe[columnX_2nd]
        alpha, self.coef_path, _ = lasso_path(train_woe_2ndX, trainY, alphas=self.lasso_list, positive=True)
    
    def LR_model(self):
        self.columnX_final = dict()
        self.LR = dict()
        for i in range(len(self.lasso_list)):
            self.columnX_final[i] = self.columnX_2nd[self.coef_path[:,i]>0]
            self.trainX = self.train_woe[self.columnX_final[i]]
            self.trainY = self.ins[self.columnY]
            self.testX = self.test_woe[self.columnX_final[i]]
            self.testY = self.oot[self.columnY]
            self.LR[i] = LogisticRegression(penalty='l2', random_state=1000, tol=1, C=10, fit_intercept = True,
                        solver='liblinear', class_weight = {0:1, 1:1}).fit(self.trainX.values, self.trainY.values)
            self.trainYScore, self.testYScore = self.mi.roc_lr(self.LR[i], self.trainX, self.trainY, self.testX, self.testY)
            
    def lift_chart(self):
        self.trainLift5, binList5 = self.mi.lift(self.trainY, self.trainYScore, bin_num = 5)
        self.testLift5, _ = self.mi.lift(self.testY, self.testYScore, 5, binList5)
        self.trainLift10, binList10 = self.mi.lift(self.trainY, self.trainYScore, bin_num = 10)
        self.testLift10, _ = self.mi.lift(self.testY, self.testYScore, 10, binList10)
        
    def plot_lift(self):
        ax1 = plt.subplot(2,1,1)
        interval1 = 0.2
        x1 = [i-interval1 for i in range(self.trainLift5.shape[0])]
        x2 = [i+interval1 for i in range(self.trainLift5.shape[0])]
        ax1.bar(x1, self.trainLift5['overdue_rate'], width=interval1*2, color='steelblue', align='center', label = 'INS')
        ax1.bar(x2, self.testLift5['overdue_rate'], width=interval1*2, color='olive', align='center', label = 'OOT')
        ax1.legend()
        plt.xlabel('Bins')
        plt.ylabel('Overdue rate')

        ax2 = plt.subplot(2,1,2)
        interval2 = 0.2
        x3 = [i-interval2 for i in range(self.trainLift10.shape[0])]
        x4 = [i+interval2 for i in range(self.trainLift10.shape[0])]
        ax2.bar(x3, self.trainLift10['overdue_rate'], width=interval2*2, color='steelblue', align='center', label = 'INS')
        ax2.bar(x4, self.testLift10['overdue_rate'], width=interval2*2, color='olive', align='center', label = 'OOT')
        ax2.legend()
        plt.xlabel('Bins')
        plt.ylabel('Overdue rate')

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=.4)
        plt.show()
            
    def run(self):
        self.assign_para()
        self.basic_inspection()
        print("Basic Inspected")
        print(self.columnX_1st)
        self.advanced_inspection()
        print("Advanced Inspected")
        print(self.columnX_2nd)
        self.features_to_bins()
        print("Features to Bin")
        self.lasso_selection()
        print("Lasso Selection")
        self.LR_model()
        print(self.columnX_final)
        print("Plot Lift Chart")
        self.lift_chart()
        self.plot_lift()
        return self.trainYScore, self.testYScore, self.LR, self.advanced_statistic
        
        
        