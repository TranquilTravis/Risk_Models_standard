import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt
import pdb
#    pdb.set_trace()

class model_inspection:
    def bins_to_list(self, bins):
        binList = list()
        for i in range(len(bins)+1):
            if i == 0:
                binList.append([0, bins[i]])
            elif i == len(bins):
                binList.append([bins[i-1], 1])
            else:
                binList.append([bins[i-1], bins[i]])
        return binList
    
    def ks(self, positive, total):
        T = total.sum()
        P = positive.sum()
        N = T - P
        posList = list()
        totalList = list()
        ksList = list()
        for i in range(positive.shape[0]):
            tBin = total.iloc[:i+1].sum()
            pBin = positive.iloc[:i+1].sum() 
            nBin = tBin - pBin
            posList.append(pBin)
            totalList.append(tBin)
            ksList.append(float(nBin)/N - float(pBin)/P)
        return posList, totalList, ksList
            
    
    def lift(self, yValue, score, bin_num=5, bins = []):
        if len(bins) == 0:
            bins = list()
            num = bin_num
            score_sort = np.sort(score)
            while num > 1 and score_sort.size>0:
                raw_bin_idx = int(score_sort.size/num)
                current_bin_value = score_sort[raw_bin_idx-1]
                score_sort = score_sort[score_sort>current_bin_value]
                bins += [current_bin_value]
                num += -1
        liftDF = pd.DataFrame(columns = ['label','score','idx'])
        liftDF['label'] = yValue.iloc[:,0].tolist()
        liftDF['score'] = score.tolist()
        idx = np.digitize(liftDF['score'], bins, right=True)
        liftDF['idx'] = idx.tolist()
        aggList = liftDF[['label','idx']].groupby(['idx']).agg(['sum','count'])
        df = pd.DataFrame()
        df['bin'] = self.bins_to_list(bins)
        df['overdue_rate'] = aggList['label']['sum']/aggList['label']['count'].astype(float)
        df['overdue'] = aggList['label']['sum']
        df['total'] = aggList['label']['count']
        df['proportion'] = df['total']/float(liftDF.shape[0])
        df['overdue_acc'], df['total_acc'], df['ks'] = self.ks(df['overdue'], df['total'])
        
        return df, bins, idx
    
    def plot_result_sample(self, trainY, train_score, testY, test_score, bins_list):
        num = len(bins_list)
        trainLift = list()
        testLift = list()
        binList = list()
        for bin_idx in range(num):
            trainLift_tmp, binList_tmp, _ = self.lift(trainY, train_score, bins_list[bin_idx])
            trainLift.append(trainLift_tmp)
            binList.append(binList_tmp)
            testLift_tmp, _, _ = self.lift(testY, test_score, bins_list[bin_idx], binList_tmp)
            testLift.append(testLift_tmp)
            
        for i in range(num):
            fig = plt.figure()
            ax1 = fig.add_subplot(2, 1, 1)
            interval = 0.2
            x1 = [i-interval+1 for i in range(trainLift[i].shape[0])]
            x2 = [i+interval+1 for i in range(trainLift[i].shape[0])]
            ax1.bar(x1, trainLift[i]['overdue_rate'], width=interval*2, color='steelblue', align='center', label = 'INS Lift')
            ax1.bar(x2, testLift[i]['overdue_rate'], width=interval*2, color='olive', align='center', label = 'OOT Lift')
            plt.xlabel('Bins')
            plt.ylabel('Overdue rate')
            ax1.legend()

            ax2 = fig.add_subplot(2, 1, 2)
            ax2.plot(range(1,bins_list[i]+1),trainLift[i]['proportion'], color='red', label='INS prop')
            ax2.plot(range(1,bins_list[i]+1),testLift[i]['proportion'], color='blue', label='OOT prop')
            plt.ylim((0, 2*1.0/bins_list[i]))
            plt.xlabel('Bins')
            plt.ylabel('Proportion')
            ax2.legend()
            
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=.4)
            
        return trainLift, testLift, binList
    
    def plot_result_verify(self, Y, Y_pred, bins_list, binList_list):
        num = len(bins_list)
        lift_LR = list()
        idx_LR = list()
        for bin_idx in range(num):
            lift_LR_tmp, _, idx_LR_tmp = self.lift(Y, Y_pred, bins_list[bin_idx], binList_list[bin_idx])
            lift_LR.append(lift_LR_tmp)
            idx_LR.append(idx_LR_tmp)
            
        fig = plt.figure()
        for i in range(num):
            ax = fig.add_subplot(num, 1, i+1)
            ax.bar(range(lift_LR[i].shape[0]), lift_LR[i]['overdue_rate'], color='steelblue', align='center', label='lift')
            ax.plot(lift_LR[i]['ks'], color='red', label='ks')
            ax.plot(lift_LR[i]['proportion'], color='blue', label='sample proportion')
            plt.xlabel('Bins')
            plt.ylabel('Overdue rate')
            plt.legend(loc=2)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=.4)
        plt.show()
        return lift_LR, idx_LR
    
    def score_bin(self, score, bins):
        idx = np.digitize(score, bins, right=True)
        return idx
    
    def roc_xgb(self, model, trainX, trainY, testX, testY):
        #testXmat = xgb.DMatrix(testX.values)
        testXmat = testX.values
        testYScore = model.predict(testXmat)
        #pdb.set_trace()
        fprTest, tprTest, thresholdsTest = metrics.roc_curve(testY, testYScore, pos_label=1)
        ksTest = (tprTest-fprTest).max()
        aucTest = metrics.auc(fprTest, tprTest)

        #trainXmat = xgb.DMatrix(trainX.values)
        trainXmat = trainX.values
        trainYScore = model.predict(trainXmat)
        fprTrain, tprTrain, thresholdsTrain = metrics.roc_curve(trainY, trainYScore, pos_label=1)
        ksTrain = (tprTrain-fprTrain).max()
        aucTrain = metrics.auc(fprTrain, tprTrain)

        plt.plot(fprTest, tprTest, color="blue")
        plt.plot(fprTrain, tprTrain, color="green")

        plt.plot([0,0.5,1],[0,0.5,1], color="orange")
        plt.xlabel('False Alarm Rate')
        plt.ylabel('Detection Rate')
        plt.title('ROC curve')
        plt.show()

        print("INS:\t\t", ksTrain, aucTrain)
        print("OOT:\t\t", ksTest, aucTest)
        
        return trainYScore, testYScore
        
    def roc_lr(self, LR, trainX, trainY, testX, testY):
        testYScore = LR.predict_proba(testX.values)[:,1]
        trainYScore = LR.predict_proba(trainX.values)[:,1]

        fprTest, tprTest, thresholdsTest = metrics.roc_curve(testY, testYScore, pos_label=1)
        fprTrain, tprTrain, thresholdsTrain = metrics.roc_curve(trainY, trainYScore, pos_label=1)

        ksTest = (tprTest-fprTest).max()
        aucTest = metrics.auc(fprTest, tprTest)

        ksTrain = (tprTrain-fprTrain).max()
        aucTrain = metrics.auc(fprTrain, tprTrain)

        plt.plot(fprTest, tprTest, color="blue", label='OOT')
        plt.plot(fprTrain, tprTrain, color="green", label='INS')

        plt.plot([0,0.5,1],[0,0.5,1], color="orange")
        plt.xlabel('False Alarm Rate')
        plt.ylabel('Detection Rate')
        plt.title('ROC curve')
        plt.legend()
        plt.show()

        print("INS:\t\t", ksTrain, aucTrain)
        print("OOT:\t\t", ksTest, aucTest)
        
        return trainYScore, testYScore
    
    def roc_nn(self, model, trainX, trainY, testX, testY):
        testYScore = model.predict(testX.values)
        trainYScore = model.predict(trainX.values)

        fprTest, tprTest, thresholdsTest = metrics.roc_curve(testY, testYScore, pos_label=1)
        fprTrain, tprTrain, thresholdsTrain = metrics.roc_curve(trainY, trainYScore, pos_label=1)

        ksTest = (tprTest-fprTest).max()
        aucTest = metrics.auc(fprTest, tprTest)

        ksTrain = (tprTrain-fprTrain).max()
        aucTrain = metrics.auc(fprTrain, tprTrain)

        plt.plot(fprTest, tprTest, color="blue")
        plt.plot(fprTrain, tprTrain, color="green")

        plt.plot([0,0.5,1],[0,0.5,1], color="orange")
        plt.xlabel('False Alarm Rate')
        plt.ylabel('Detection Rate')
        plt.title('ROC curve')
        plt.show()

        print("INS:\t\t", ksTrain, aucTrain)
        print("OOT:\t\t", ksTest, aucTest)
        
        return trainYScore, testYScore
    
    def single_feature_basic_summary(self, featureS, psiS):
        feature1List = featureS[['nan_ratio', 'zero_rate', 'ks', 'auc']].tolist()
        feature2List = psiS[['psi', 'iv_train', 'iv_test', 'method']].tolist()
        return feature1List+feature2List
        
    def single_feature_bin_summary(self, psiS, featureD, bin_stat):
        binList = psiS['bins']
        woeList = featureD['bin_woe']
        briList = psiS['trainP']
        broList = psiS['testP']
        ttiList = psiS['trainT']
        ttoList = psiS['testT']
        for i in range(len(binList)+1):
            if i == 0:
                bin_range = [-np.inf, binList[i]]
            elif i == len(binList):
                bin_range = [binList[i-1], np.inf]
            else:
                bin_range = [binList[i-1], binList[i]]
            bin_stat.loc[bin_stat.shape[0]] = [psiS.name, i, bin_range, woeList[i], briList[i], broList[i], ttiList[i], ttoList[i]]
        return bin_stat
    
    def feature_summary(self, feature_stat, psi_stat, feature_dict, columnFeature):
        basic_stat = pd.DataFrame(index = columnFeature, columns = ['nan_rate', 'zero_rate', 'ks', 'auc', 'psi', 'iv_ins', 'iv_oot', 'nan_method'])
        bin_stat = pd.DataFrame(columns = ['feature', 'bin_order', 'bin', 'woe', 'br_ins', 'br_oot', 'total_ins', 'total_oot'])
        for col in columnFeature:
            featureS = feature_stat.loc[col]
            psiS = psi_stat.loc[col]
            featureD = feature_dict.loc[col]
            basic_stat.loc[col] = self.single_feature_basic_summary(featureS, psiS)
            bin_stat = self.single_feature_bin_summary(psiS, featureD, bin_stat)
        bin_stat = bin_stat.set_index(['feature', 'bin_order'])
        return basic_stat, bin_stat
            
        


