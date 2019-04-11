import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import general_func as gf
import pdb
#    pdb.set_trace()

class binning:
    def bin_split(self, feature, initial_num=30, min_split_sample=10):
        """
        (Inner function) 
        split feature values to bins
        Input:
            feature: single column dataframe
            initial_num: default initial number of bins
            min_split_sample: minimum number of samples to split a bin
        return: 
            bins: the list of split point
        """
        bins = [] # empty bin
        num_bins = initial_num # current number of bins to split
        trans_feature = feature.sort_values(ascending=True)
    
        while num_bins > 1 and trans_feature.shape[0]>min_split_sample:
            raw_bin_idx = int(trans_feature.shape[0]/num_bins)
            current_bin_value = trans_feature.iloc[raw_bin_idx-1]
            trans_feature = trans_feature[trans_feature>current_bin_value]
            if trans_feature.shape[0]>min_split_sample:
                bins += [current_bin_value]
            num_bins += -1
        return bins
    
    def bin_feature(self, feature, label, bins):
        """
        (Inner function) called by self.advanced_statistic()
        assign feature value with bin index according to bins
        Input:
            feature: single column dataframe
            label: label dataframe
            bins: the list of split point
        Return:
            feature_bin: dataframe (feature, label, belonging index)
        """
        col = ['feature', 'label', 'idx']
        feature_bin = pd.DataFrame(columns = col)
        feature_bin['feature'] = feature
        feature_bin['label'] = label
        idx = np.digitize(feature_bin['feature'], bins, right=True)
        feature_bin['idx'] = idx
        return feature_bin
    
    def base_statistic(self, df):
        """
        (Inner function)
        total positive, total negative, total number
        """
        positive = df[df['label']==1].shape[0]
        negative = df[df['label']==0].shape[0]
        return positive, negative, positive+negative
    
    def bin_statistic(self, binOne, binTwo, one, two):
        """
        (Inner function)
        general function of (binOne/one-binTwo/two)*(log(binOne/one)-log(binTwo/two))
        for example:
            iv bin: binOne=binBad, binTwo=binGood, one=badTotal, two=goodTotal
            psi bin: binOne=actualBinTotal, binTwo=expectBinTotal, one=actualTotal, two=expectTotal
            tsi bin: binOne=actualBinBad, binTwo=expectBinBad, one=actualBinTotal, two=expectBinTotal
        """
        if one == 0 or two == 0:
            return 0
        else:
            one_rate = float(binOne)/one
            two_rate = float(binTwo)/two
        if one_rate == 0:
            tmp = 3
        elif two_rate == 0:
            tmp = -3
        else:
            tmp = np.log(one_rate)-np.log(two_rate)
        binStat = (one_rate - two_rate) * tmp
        return binStat
    
    def woe(self, binBad, binGood, bad, good):
        """
        (Inner function) 
        compute woe of a bin
        """
        bad_rate = (float(binBad)/bad)[0]
        good_rate = (float(binGood)/good)[0]
        if bad_rate == 0:
            woe = 3
        elif good_rate == 0:
            woe = -3
        else:
            woe = np.log(bad_rate)-np.log(good_rate)
        return woe
    
    def advanced_statistic(self, train_feature, train_label, test_feature, test_label, bins):
        """
        (Inner function) called by self.all_feature_psi
        return iv, psi, tsi
        Input: 
            train_feature, test_feature: single column dataframe
            tain_label, test_label: label dataframe
            bins: the list of split point
        Return:
            iv_train: iv for train feature
            iv_test: iv for test feature
            psi: population stability index
            tsi: targeted stability index
            train_bin_p, test_bin_p: the list of positive samples in each bin
            train_bin, test_bin: the list of all samples in each bin
        """
        trainX = train_feature
        testX = test_feature
        train = self.bin_feature(trainX, train_label, bins)
        test = self.bin_feature(testX, test_label, bins)
        trainP, trainN, trainT = self.base_statistic(train)
        testP, testN, testT = self.base_statistic(test)
        iv_train = 0
        iv_test = 0
        psi = 0
        tsi = 0
        train_bin_p = list()
        train_bin =list()
        test_bin_p = list()
        test_bin = list()
        train_p_num = list() #test
        for theidx in range(len(bins)+1):
            trainBinP, trainBinN, trainBinT = self.base_statistic(train[train['idx']==theidx])
            testBinP, testBinN, testBinT = self.base_statistic(test[test['idx']==theidx])
            train_bin_p += [trainBinP/float(trainBinT)]
            train_p_num += [trainBinP] # test
            train_bin += [trainBinT] 
            test_bin_p += [0 if testBinT==0 else testBinP/float(testBinT)]
            test_bin += [testBinT]
            ivBinTrain = self.bin_statistic(trainBinP, trainBinN, trainP, trainN)
            ivBinTest = self.bin_statistic(testBinP, testBinN, testP, testN)
            #pdb.set_trace()
            psiBin = self.bin_statistic(testBinT, trainBinT, testT, trainT)
            tsiBin = self.bin_statistic(testBinP, trainBinP, testBinT, trainBinT)
            
            iv_train += ivBinTrain
            iv_test += ivBinTest
            psi += psiBin
            tsi += tsiBin
        return iv_train, iv_test, psi, tsi, train_bin_p, test_bin_p, train_bin, test_bin
        
    def bin_mat(self, feature, bins):
        """
        (Inner function) called by self.bin_grouping
        construct the feature binning matrix with (bin range) & (positive labels in bin) & (negative labels in bin)
        Input:
            feature: single column dataframe
            bins: the list of split point
        Return:
            df_mat: feature binning matrix
        """
        df_mat = pd.DataFrame(columns=['bin_range', 'bad', 'good'])
        for idx in np.sort(feature['idx'].unique()):
            if idx == 0:
                binRange = [-np.inf, bins[0]]
            elif idx == len(bins):
                binRange = [bins[idx-1], np.inf]
            else:
                binRange = [bins[idx-1], bins[idx]] 
            good = feature[(feature['idx']==idx)&(feature['label']==0)].shape[0]
            bad = feature[(feature['idx']==idx)&(feature['label']==1)].shape[0]
            df_mat.loc[idx] = [binRange, bad, good]
            
        return df_mat

    def merge_two_bins(self, bin_mat, bin_ranges, merge_idx):
        bin_mat[merge_idx] = bin_mat[merge_idx] + bin_mat[merge_idx+1]
        bin_mat = np.delete(bin_mat, merge_idx+1, 0)
        bin_ranges[merge_idx] = [bin_ranges[merge_idx][0], bin_ranges[merge_idx+1][1]]
        bin_ranges = np.delete(bin_ranges, merge_idx+1, 0)
        return bin_mat, bin_ranges
    
    def bin_bad_rate(self, x):
        """
        (Inner function) 
        compute bin bad rate
        """
        return float(x[0])/(x[0]+x[1])
    
    def bin_bad_rate_diff(self, bin_mat, if_mono):
        """
        (Inner function) called by self.grouping_bad_rate
        compute bad rate difference for adjacent bins
        """
        bad_rate = np.apply_along_axis(self.bin_bad_rate, 1, bin_mat)
        if if_mono:
            sign = np.sign(bad_rate[-1]-bad_rate[0])
            diff = np.diff(bad_rate)*sign
        else:
            diff = np.absolute(np.diff(bad_rate))
        return diff
        
    def grouping_bad_rate(self, bin_mat, bin_ranges, if_mono = True, stop_condition=0.01):
        """
        (Inner function) called by self.bin_grouping
        this is the bad_rate grouping method
        Input:
            bin_mat: dataframe [... [bin_i negatives, bin_i positives] ... ]
            bin_ranges: [... [bin_i start, bin_i end] ...]
            if_mono: if bins with monotonous bad rate
            stop_condition: the grouping stop condition
        Return:
            bin_ranges: return grouped [... [bin_i start, bin_i end] ...]
        """
        diff = self.bin_bad_rate_diff(bin_mat, if_mono)    
        while diff.min() <= stop_condition and diff.shape[0]>1:
            # merge
            idx = diff.argmin()
            bin_mat, bin_ranges = self.merge_two_bins(bin_mat, bin_ranges, idx)
            diff = self.bin_bad_rate_diff(bin_mat, if_mono) 
        if diff.min() <= stop_condition and diff.shape[0] == 1:
            return np.array([]), np.array([[-np.inf, np.inf]])
        return bin_mat, bin_ranges
    
    def chi2(self, badBin1, goodBin1, badBin2, goodBin2):
        """
        """
        total = badBin1+goodBin1+badBin2+goodBin2
        expected1_bad = float(badBin1+badBin2)/total*(badBin1+goodBin1)
        expected2_bad = float(badBin1+badBin2)/total*(badBin2+goodBin2)
        expected1_good = float(goodBin1+goodBin2)/total*(badBin1+goodBin1)
        expected2_good = float(goodBin1+goodBin2)/total*(badBin2+goodBin2)
        chi2 = (badBin1-expected1_bad)**2/expected1_bad + (badBin2-expected2_bad)**2/expected2_bad + \
                (goodBin1-expected1_good)**2/expected1_good + (goodBin2-expected2_good)**2/expected2_good
        return chi2
    
    def all_bins_chi2(self, bin_mat):
        """
        Input:
            bin_mat: numpy array (bad & good)
        """
        chi2List = list()
        for i in range(bin_mat.shape[0]-1): 
            chi2 = self.chi2(bin_mat[i,0], bin_mat[i,1], bin_mat[i+1,0], bin_mat[i+1,1])
            chi2List.append(chi2)
        return np.array(chi2List)
    
    def grouping_chi2(self, bin_mat, bin_ranges, if_mono = True, stop_condition = 3.841):
        """
        (Inner function) called by self.bin_grouping
        this is the chi-square grouping method
        
        """
        # if require mono bad rate, join all bins without mono bad rate
        if if_mono == True:
            bin_mat, bin_ranges = self.grouping_bad_rate(bin_mat, bin_ranges, True, 0.0)
            if bin_mat.shape[0] == 0:
                return bin_mat, bin_ranges
           
        chi2List = self.all_bins_chi2(bin_mat)
        while chi2List.min() <= stop_condition and chi2List.size > 1:
            idx = chi2List.argmin()
            bin_mat, bin_ranges = self.merge_two_bins(bin_mat, bin_ranges, idx)
            chi2List = self.all_bins_chi2(bin_mat)
        if chi2List.min() <= stop_condition and chi2List.size == 1:
            return np.array([]), np.array([[-np.inf, np.inf]])
        return bin_mat, bin_ranges
    
    def bin_grouping(self, feature, label, bins, method='bad_rate', if_mono=True, stop_condition=0.01):
        """
        (Inner function) called by self.all_feature_psi
        General bin grouping method
        Input:
            feature: single column dataframe
            label: label dataframe
            method: the method of grouping bins
            if_mono: if bins with monotonous bad rate
            stop_condition: the grouping stop condition
        return:
            bin_ranges: [... [bin_i start, bin_i end] ...]
            groupingBins: the list of split point
        """
        df = self.bin_feature(feature, label, bins)
        df_mat = self.bin_mat(df, bins)
        bin_ranges = df_mat['bin_range'].values
        bin_mat = df_mat[['bad','good']].values
        if method == 'bad_rate':
            _, bin_ranges = self.grouping_bad_rate(bin_mat, bin_ranges, if_mono, stop_condition)
        elif method == 'chi2':
            _, bin_ranges = self.grouping_chi2(bin_mat, bin_ranges, if_mono, stop_condition)

        bins = [x[1] for x in bin_ranges if x[1]!=np.inf]
        return bin_ranges, bins
    
    def all_feature_psi(self, train_features, train_label,
                        test_features, test_label, feature_stat,
                        initial_num=10, min_split_sample=10, method='bad_rate',
                        if_mono=True,stop_condition=0.01):
        """
        (API function)
        the second round feature inspection
        Input: 
            train_features, test_features: features dataframe
            train_label, test_label: label dataframe
            fill_nan_method: use for self.fill_nan function
            initial_num, min_split_sample: use for self.bin_split function
            method, if_mono, stop_condition: use for self.bin_grouping function
        Return:
            statistic: second round feature statistics
        """
        statistic = pd.DataFrame(index = train_features.columns,
                                 columns = ['bins', 'iv_train', 'iv_test', 'psi', 'tsi', 'trainP', 'testP', 'trainT', 'testT','method'])
        for f in train_features.columns:
            fill_nan_method = feature_stat.loc[f,'method']
            encode_dict = feature_stat.loc[f,'encode_dict']
            train_feature = train_features[f].copy()
            test_feature = test_features[f].copy()
            train_feature.replace(encode_dict, inplace=True)
            test_feature.replace(encode_dict, inplace=True)
            bins = self.bin_split(train_feature, initial_num, min_split_sample)
            if len(bins) > 0:
                _, bins = self.bin_grouping(train_feature, train_label, bins, method, if_mono, stop_condition)
            if len(bins) == 0:
                iv_train, iv_test, psi, tsi, train_bin_p, test_bin_p, train_bin, test_bin \
                = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            else:
                iv_train, iv_test, psi, tsi, train_bin_p, test_bin_p, train_bin, test_bin \
                                    = self.advanced_statistic(
                                    train_feature, train_label,
                                    test_feature, test_label, bins)
            statistic.loc[f] = [bins, iv_train, iv_test, psi, tsi, train_bin_p, test_bin_p, train_bin, test_bin, fill_nan_method]
        return statistic
    
    def feature_to_bins(self, data, col, columnY, bins, encode_dict, bin_woe = list()):
        feature = data[col].copy()
        feature.replace(encode_dict, inplace=True)
        idx = np.digitize(feature, bins, right=True)
        df = pd.DataFrame(columns=['feature', 'idx', 'label', 'woe'])
        df['feature'] = data[col]
        df['label'] = data[columnY]
        df['idx'] = idx
        good = (data[columnY]==0).sum()
        bad = (data[columnY]==1).sum()
        bin_order = list()
        if len(bin_woe) == 0:
            flag = 0
        else:
            flag = 1
        for i in range(len(bins)+1):
            if flag == 0:
                binBad = df[df['idx']==i]['label'].sum()
                binGood = (df[df['idx']==i]['label']==0).sum()
                currentWOE = self.woe(binBad, binGood, bad, good)
                bin_woe.append(currentWOE)
            else:
                currentWOE = bin_woe[i]
            df.loc[df['idx']==i,'woe'] = currentWOE
            bin_order.append(i)
        return df, bin_order, bin_woe
    
    def all_features_to_bins(self, features, columnX, columnY, psiDF, feature_stat, woe = pd.DataFrame()):
        """
        API function
        """
        feature_dict = pd.DataFrame(index = columnX, columns = ['bins', 'bin_order', 'bin_woe', 'bin_len'])
        feature_woe = pd.DataFrame()
        feature_bin = pd.DataFrame()
        feature_woe[columnY] = features[columnY]
        feature_bin[columnY] = features[columnY]
        for col in columnX:
            bins = psiDF.loc[col,'bins']
            encode_dict = feature_stat.loc[col,'encode_dict']
            if woe.shape[0] > 0:
                bin_woe = woe.loc[col]
                featureDF, bin_order, _ = self.feature_to_bins(features, col, columnY, bins, encode_dict, bin_woe)
            else:
                bin_woe = list()
                featureDF, bin_order, bin_woe = self.feature_to_bins(features, col, columnY, bins, encode_dict, bin_woe)
            feature_woe[col] = featureDF['woe']
            feature_bin[col] = featureDF['idx']
            feature_dict.loc[col] = [bins, bin_order, bin_woe, len(bin_order)]
        return feature_woe, feature_bin, feature_dict
    
    def all_features_only_woe(self, features, columnX, woe):
        features_woe = pd.DataFrame()
        for col in columnX:
            bins = woe.loc[col,'bins']
            fill_nan_method = woe.loc[col,'method']
            bin_woe = woe.loc[col, 'bin_woe']
            feature = gf.fill_nan(fill_nan_method, features[col])
            idx = np.digitize(feature, bins, right=True)
            df = pd.DataFrame(columns=['feature', 'idx', 'woe'])
            df['feature'] = feature
            df['idx'] = idx
            for i in range(len(bins)+1):
                currentWOE = bin_woe[i]
                df.loc[df['idx']==i,'woe'] = currentWOE
            features_woe[col] = df['woe']
        return features_woe
            
    
    def all_woe_features_to_bins(self, feature_bin):
        """
        API function
        """
        return feature_bin.apply(lambda x:x.astype(float)/x.max(), axis = 1)
            
    
    def bin_plot(self, df):
        bad_rate_list = list()
        feature_name = df.columns[0]
        for i in np.sort(df[feature_name].unique()):
            currentBin = df[df[feature_name]==i]
            bad_rate_list += [float(currentBin['label'].sum())/currentBin['label'].count()]
        plt.bar(range(df[feature_name].unique().shape[0]), bad_rate_list)
        plt.show()
        return bad_rate_list
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        