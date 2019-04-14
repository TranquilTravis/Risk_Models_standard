import pandas as pd
import numpy as np
import pdb
#    pdb.set_trace()

class feature_derive:
    def generate_tree_structure(self, estimator, feature_names=None):
        """
        函数说明：层次遍历，获取叶子节点的决策路径
        参数：
            estimator--训练得到的树结构体，sklearn.tree.tree_.Tree实例
            feature_names--特征名列表，list结构
        return：generated_bin，结构为：[path_leaf1, path_leaf2...], 
            每个path_leafi为从根节点到相应叶子节点的路径，[node1,node2...]
            每个node为元组(node_id, feature_id, feature_name，symbol，threshold)
                其中symbol为boolean，0表示left节点，即<=，1表示right节点的>
        """
        n_nodes = estimator.tree_.node_count
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold

        decision_path = {}
        generated_bin = []
        stack = [0]
        decision_path[0] = []
        while len(stack)>0:
            curr_id = stack.pop()

            left = estimator.tree_.children_left[curr_id]
            right = estimator.tree_.children_right[curr_id]

            if right<0 and left<0: #找到叶子节点
                generated_bin.append(decision_path[curr_id])
                continue

            if right>0:
                stack.append(right)
            if left>0:
                stack.append(left)
            if feature_names:
                left_tuple = (left, feature[curr_id], feature_names[feature[curr_id]], 0, threshold[curr_id])
                right_tuple = (right, feature[curr_id], feature_names[feature[curr_id]], 1, threshold[curr_id])
            else:
                left_tuple = (left, feature[curr_id], 0, threshold[curr_id])
                right_tuple = (right, feature[curr_id], 1, threshold[curr_id])
            decision_path[left] = decision_path[curr_id].copy()
            decision_path[left].append(left_tuple)
            decision_path[right] = decision_path[curr_id].copy()
            decision_path[right].append(right_tuple)

        return generated_bin
    
    # 树结构体主要元素
    def get_tree_structure(self, estimator):
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        
        
    #可视化树结构
    def plot_tree(self, estimator, result_dir=None, result_file=None, features=None, targets=None):
        dot_data = tree.export_graphviz(estimator, class_names=targets, node_ids=True,
                                        feature_names=features, out_file=None, filled=True)
        graph = graphviz.Source(dot_data)
        if result_file:
            graph.render(result_file, format='png', directory=result_dir)
        return graph
    
    
    # 层次遍历，判断每个节点的深度&是否为叶子节点
    def get_depth_isleaf(self, estimator):
        n_nodes = estimator.tree_.node_count

        depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaf = np.zeros(shape=n_nodes, dtype=np.int64)

        stack=[(0, -1)]
        while len(stack)>0:
            curr_id, parent_depth=stack.pop()

            depth[curr_id] = parent_depth+1


            left = estimator.tree_.children_left[curr_id]
            right = estimator.tree_.children_right[curr_id]
            if right>0:
                stack.append((right, depth[curr_id]))
            if left>0:
                stack.append((left, depth[curr_id]))
            if right<0 and left<0:
                is_leaf[curr_id] = True
            else:
                is_leaf[curr_id] = False
        return bunch.Bunch(depth=depth, is_leaf=is_leaf)
    
    def all_feature_derive(self, features, columnX, columnY, model, flag, feature_dict = pd.DataFrame()):
        """
        API function
        flag: 0 for models without estimator method
              1 for models with estimator method
        """
        if feature_dict.shape[0]==0:
            feature_dict = pd.DataFrame(columns = ['bins', 'bin_woe', 'bin_len'])
            has_feature_dict = 0
        else:
            has_feature_dict = 1
        dataX = features[columnX]
        dataY = features[columnY]
        if flag == 0:
            feature_derived = model.apply(dataX.values)
        elif flag == 1:
            feature_derived = np.empty([dataX.shape[0], 0])
            for i, est in enumerate(model.estimators_):
                tmp = est.apply(dataX.values)
                feature_derived = np.column_stack([feature_derived, tmp])
        columnX_derive = ['f{0}'.format(i) for i in range(feature_derived.shape[1])]
        feature_woe = pd.DataFrame(index = features.index)
        feature_bin = pd.DataFrame(data = feature_derived, columns = columnX_derive, index = features.index)
#         feature_woe[columnY] = features[columnY]
#         feature_bin[columnY] = features[columnY]
        for i, col in enumerate(columnX_derive):
            if has_feature_dict == 0:
                bins = np.unique(feature_derived[i])
                featureDF, bin_order, bin_woe = self.feature_to_bins(feature_bin[col], dataY, bins)
                feature_woe[col] = featureDF['woe']
                feature_dict.loc[col] = [bin_order, bin_woe, len(bin_order)]
            else:
                tmp_dict = feature_dict.loc['f{0}'.format(i)]
                featureDF = self.feature_to_bins_with_dict(feature_bin[col], tmp_dict)
                feature_woe[col] = featureDF['woe']
                
        return feature_woe, feature_bin, feature_dict
    
    def feature_to_bins(self, dataX, dataY, bins):
        df = pd.DataFrame(columns=['feature', 'label', 'woe'])
        df['feature'] = dataX
        df['label'] = dataY
        good = (dataY==0).sum()
        bad = (dataY==1).sum()
        bin_order = list()
        bin_woe = list()
        for i in dataX.unique():
            binBad = df[df['feature']==i]['label'].sum()
            binGood = (df[df['feature']==i]['label']==0).sum()
            currentWOE = self.woe(binBad, binGood, bad, good)
            bin_woe.append(currentWOE)
            bin_order.append(i)
            df.loc[df['feature']==i,'woe'] = currentWOE       
        return df, bin_order, bin_woe
    
    def feature_to_bins_with_dict(self, dataX, f_dict):
        df = pd.DataFrame(columns=['feature', 'woe'])
        df['feature'] = dataX
        for i, value in enumerate(f_dict.bins):
            df.loc[df['feature']==value,'woe'] = f_dict.bin_woe[i]
        return df
    
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

