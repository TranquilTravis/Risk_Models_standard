import pandas as pd
import numpy as np
from sklearn import metrics
import general_func as gf
from imp import reload
import matplotlib.pyplot as plt
import pdb
#    pdb.set_trace()

class feature_inspection:
    """
    feature: pandas series
    """
        
    def nan_ratio(self, feature):
        #return float(np.isnan(feature).sum())/feature.size
        return float(feature.isnull().sum())/feature.size

    def zero_ratio(self, feature):
        #return float((feature==0).sum())/(~np.isnan(feature)).sum()
        return float((feature==0).sum())/(~feature.isnull()).sum()
    
    def inplace_encoding(self, feature, label):
        feature_df_org = pd.DataFrame(columns = ['feature', 'label'])
        feature_df_org['feature'] = feature
        feature_df_org['label'] = label
        feature_df = feature_df_org.copy()
        feature_df = feature_df[~feature_df['feature'].isnull()]
        unique_values = feature_df['feature'].unique()
        br_df = pd.DataFrame(index = unique_values, columns = ['bad_rate', 'ranks'])
        for v in unique_values:
            tmp_df = feature_df[feature_df['feature']==v]
            br_df.loc[v,'bad_rate'] = tmp_df['label'].sum()/float(tmp_df.shape[0])
        br_df.sort_values(by=['bad_rate'], inplace=True)
        br_df['ranks'] = range(br_df.shape[0])
        feature_dict = dict()
        for v in unique_values:
            feature_dict[v] = br_df.loc[v].ranks
        feature_df_org['feature'].replace(feature_dict, inplace=True)
        return feature_df_org['feature'], feature_dict
        
    
    def ks_auc(self, feature, label, fill_nan_method_list):
        """
        API function & Inner function
        """
        if len(fill_nan_method_list) == 0:
            fill_nan_method_list = ['minus', 'plus', 'mean', 'max', 'min']
            
        ks_dict = dict()
        auc_dict = dict()
        value_dict = dict()
        encode_dict = dict()
        if feature.dtypes == np.object:
            feature, encode_dict = self.inplace_encoding(feature, label)
       
        for mtd in fill_nan_method_list:
            new_feature, v = gf.fill_nan(mtd, feature)
            fpr, tpr, thresholds = metrics.roc_curve(label, new_feature, pos_label=1)
            ks_dict[mtd] = abs(tpr-fpr).max()
            auc_dict[mtd] = max(metrics.auc(fpr, tpr),1-metrics.auc(fpr, tpr))
            value_dict[mtd] = v
#        method = max(ks_dict, key=ks_dict.get)
        method = max(auc_dict, key=auc_dict.get)
        ks = ks_dict[method]
        auc = auc_dict[method]
        encode_dict[np.nan] = value_dict[method]
        return ks, auc, method, encode_dict
    
    def precision_recall(self, new_feature, label):
        total_num = new_feature.shape[0]
        num_threshold_min = total_num*0.01
        num_threshold_max = total_num*0.3
        new_df = pd.DataFrame(columns = ['feature', 'label'])
        new_df['feature'] = new_feature
        new_df['label'] = label
        thresholds = new_feature.unique()
        precision_dict = dict()
        precision_set_dict = dict()
        count_dict = dict()
        for th in thresholds:
            tmp_df_left = new_df[new_df['feature']<=th]
            tmp_df_right = new_df[new_df['feature']>th]
            if (tmp_df_left.shape[0]>num_threshold_min and tmp_df_left.shape[0]<num_threshold_max) or \
               (tmp_df_right.shape[0]>num_threshold_min and tmp_df_right.shape[0]<num_threshold_max):
                precision_left = tmp_df_left['label'].sum()/float(tmp_df_left.shape[0])
                precision_right = tmp_df_right['label'].sum()/float(tmp_df_right.shape[0])
                count_dict[th] = [tmp_df_left.shape[0], tmp_df_right.shape[0]]
                precision_dict[th] = max(precision_left,precision_right)
                precision_set_dict[th] = [precision_left, precision_right]
        if len(precision_dict) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            th_max = max(precision_dict, key=precision_dict.get)
        return th_max, count_dict[th_max][0], count_dict[th_max][1], precision_set_dict[th_max][0], precision_set_dict[th_max][1]
    
    def all_feature_inspection(self, df, columnX, columnY, fill_nan_method_list = list()):
        """
        API function
        """
        label = df[columnY]
        statistic = pd.DataFrame(index = columnX, columns = ['nan_ratio', 'zero_rate', 'ks', 'auc', 'method', 'encode_dict'])
        for col in columnX:
            feature = df[col]
            nan_ratio = self.nan_ratio(feature)
            if nan_ratio == 1:
                statistic.loc[col] = [1, 0, 0, 0.5, np.nan, {np.nan:''}]
            else:
                zero_rate = self.zero_ratio(feature)
                ks, auc, method, encode_dict = self.ks_auc(feature, label, fill_nan_method_list)
                statistic.loc[col] = [nan_ratio, zero_rate, ks, auc, method, encode_dict]
        return statistic
    
    def all_feature_policy(self, df, columnX, columnY, feature_stat):
        """
        API function
        """
        label = df[columnY]
        statistic = pd.DataFrame(index = columnX, columns = ['threshold', 'count_left', 'count_right', 'bad_rate_left', 'bad_rate_right'])
        for col in columnX:
            print(col)
            nan_ratio = feature_stat.loc[col]['nan_ratio']
            if nan_ratio == 1:
                statistic.loc[col] = [np.nan, np.nan, np.nan, np.nan, np.nan]
            else:
                feature = df[col]
                feature_dict = feature_stat.loc[col]['encode_dict']
                new_feature = feature.replace(feature_dict)
                threshold, count_left, count_right, bad_rate_left, bad_rate_right = self.precision_recall(new_feature, label)
                statistic.loc[col] = [threshold, count_left, count_right, bad_rate_left, bad_rate_right]
        return statistic
    
    def all_feature_fillin_nan(self, df, columnX, statistic):
        """
        API function
        """
        new_df = df.copy()
        for col in columnX:
            feature = df[col]
            new_df[col] = feature.fillna(statistic.loc[col]['encode_dict'][np.nan])
        return new_df    
            
        
        
        
        
        
        
        


