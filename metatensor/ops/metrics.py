import abc
import numpy as np
from ..core.node import Node

"""
TP: true positive (ground_truth=True, predict=True, ground_truth==predict)
TN: true negative (ground_truth=False, predict=False, ground_truth==predict)
FP: false positive (ground_truth=False, predict=True, ground_truth!=predict)
FN: false negative (ground_truth=True, predict=False, ground_truth!=predict)
"""

class Metrics(Node):
    """
    Abstract class for metrics nodes.
    """
    def __init__(self, *parents, **kargs):
        kargs['need_save'] = kargs.get('need_save', False)
        Node.__init__(self, *parents, **kargs)
        self.init()
    
    def reset(self):
        self.reset_value()
        self.init()
    
    @abc.abstractmethod
    def value_str(self):
        return "{}: {:.4f}; ".format(self.__class__.__name__, self.value)

    @abc.abstractmethod
    def init(self):
        pass

    def get_jacobi(self):
        # Metrics nodes are not allowed to go backpropagation
        raise NotImplementedError()

    @abc.abstractmethod
    def prob_to_label(prob, thresholds=0.5):
        if prob.shape[0] > 1:
            # multi-class classification
            labels = np.zeros((prob.shape[0], 1))
            labels[np.argmax(prob, axis=0)] = 1
        else:
            # classify results according to thresholds
            labels = np.where(prob < thresholds, 0, 1)
        return labels

class Accuray(Metrics):
    """
    Evaluate accuracy. Acc = (TP + TN) / (TP + TN + FP + FN)
    """
    def __init__(self, *parents, **kargs):
        # parents[0] is output, parents[1] is ground_truth
        Metrics.__init__(self, *parents, **kargs)

    def init(self):
        self.correct_num = 0
        self.total_num = 0
    
    def compute(self):
        pred = Metrics.prob_to_label(self.parents[0].value)
        ground_truth = self.parents[1].value

        assert len(pred) == len(ground_truth)

        self.correct_num += np.sum(pred == ground_truth)
        self.total_num += len(pred)

        self.value = 0
        if self.total_num != 0:
            self.value = float(self.correct_num) / self.total_num

class Precision(Metrics):
    """
    Evaluate precision. Pre = TP / (TP + FP) = correctly predicted as True / all True predictions.
    """
    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)
    
    def init(self):
        self.num_positive_predict = 0
        self.num_positive_ground_truth_predict = 0
    
    def compute(self):
        pred = Metrics.prob_to_label(self.parents[0].value)
        ground_truth = self.parents[1].value
        self.num_positive_predict += np.sum(pred == 1)

        self.num_positive_ground_truth_predict += np.sum(np.multiply(pred == 1, pred == ground_truth))

        self.value = 0
        if self.num_positive_predict != 0:
            self.value = float(self.num_positive_ground_truth_predict) / self.num_positive_predict

class Recall(Metrics):
    """
    Evaluate true positive rate(TPR). Recall = TP / (TP + FN) = correctly predicted as True / all True samples.
    """
    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)
    
    def init(self):
        self.num_positive_ground_truth = 0
        self.num_positive_ground_truth_predict = 0
    
    def compute(self):
        pred = Metrics.prob_to_label(self.parents[0].value)
        ground_truth = self.parents[1].value

        self.num_positive_ground_truth += np.sum(ground_truth == 1)

        self.num_positive_ground_truth_predict += np.sum(np.multiply(pred == 1, pred == ground_truth))

        self.value = 0
        if self.num_positive_ground_truth != 0:
            self.value = float(self.num_positive_ground_truth_predict) / self.num_positive_ground_truth

class ROC(Metrics):
    """
    Evaluate receiver operating characteristic curve.
    """
    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)
    
    def init(self):
        self.count = 100 # num of threshold samples
        self.num_ground_truth_positive = 0
        self.num_ground_truth_negative = 0
        self.num_true_positive = np.array([0] * self.count)
        self.num_false_positive = np.array([0] * self.count)
        self.TPR = np.array([0] * self.count)
        self.FPR = np.array([0] * self.count)
    
    def compute(self):
        prob = self.parents[0].value
        ground_truth = self.parents[1].value
        self.num_ground_truth_positive = np.sum(ground_truth==1)
        self.num_ground_truth_negative = np.sum(ground_truth==-1)

        thresholds = list(np.linspace(0.0, 1.0, num=self.count, endpoint=False))

        for i in range(0, self.count):
            pred = Metrics.prob_to_label(prob, thresholds[i])
            self.num_true_positive[i] += np.sum(np.multiply(pred == 1, pred == ground_truth))
            self.num_false_positive[i] += np.sum(np.multiply(pred == 1, pred != ground_truth))
        
        # compute TPR, FPR
        if self.num_ground_truth_positive != 0 and self.num_ground_truth_negative != 0:
            self.TPR = self.num_true_positive / self.num_ground_truth_positive
            self.FPR = self.num_false_positive / self.num_ground_truth_negative
    
    def value_str(self):
        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # plt.ylim(0, 1)
        # plt.xlim(0, 1)
        # plt.plot(self.fpr, self.tpr)
        # plt.show()
        return ' '

class ROC_AUC(Metrics):
    """
    Area under ROC curve. AUC = Prob(Prob[positive] > prob[negative])
    """
    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)
    
    def init(self):
        self.positive_ground_truth = [] # all positive samples
        self.negative_ground_truth = [] # all negative samples

    def compute(self):
        prob = self.parents[0].value
        ground_truth = self.parents[1].value

        # Warning: assume ground truth is one element
        if ground_truth[0, 0] == 1:
            self.positive_ground_truth.append(prob) 
        else:
            self.negative_ground_truth.append(prob)
        
        self.total = len(self.positive_ground_truth) * len(self.negative_ground_truth)

    def value_str(self):
        count = 0 # num of P[positive] > P[negative]

        # go through (positive sample, negative sample) pairs
        for p in self.positive_ground_truth:
            for n in self.negative_ground_truth:

                if p > n:
                    count += 1

        self.value = float(count) / self.total # AUC
        return "{}: {:.4f}".format(self.__class__.__name__, self.value)

