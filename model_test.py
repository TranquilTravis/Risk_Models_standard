import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import general_func as gf
import pdb
#    pdb.set_trace()

class model_test:
    
    def all_features_to_bins_noY(self, features, columnX, stat, woe):
        """
        API function
        """
        feature_woe = pd.DataFrame()
        for col in columnX:
            bins = stat.loc[col,'bins']
            bin_woe = woe.loc[col]
            featureDF = self.feature_to_bins(features, col, bins, bin_woe)
            feature_woe[col] = featureDF['woe']
        return feature_woe
    
    def feature_to_bins(self, data, col, bins, bin_woe):
        idx = np.digitize(data[col], bins, right=True)
        df = pd.DataFrame(columns=['feature', 'idx', 'woe'])
        df['feature'] = data[col]
        df['idx'] = idx
        bin_order = list()
        for i in range(len(bins)+1):
            currentWOE = bin_woe[i]
            df.loc[df['idx']==i,'woe'] = currentWOE
            bin_order.append(i)
        return df