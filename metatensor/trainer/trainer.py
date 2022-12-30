import numpy as np
import abc
import time

class Trainer(object):
    """
    Trainer.
    """
    def __init__(self, 
                inputs, 
                labels, 
                loss, 
                optimzer, 
                num_epochs, 
                batch_size=8, 
                eval_on_train=False, 
                metrics_ops=None, 
                *args, **kargs):
        self.inputs = inputs # input nodes, this is a dictionary which key is the name of input node
        self.labels = labels # label nodes
        self.loss = loss # loss function
        self.optimzer = optimzer
        self.num_epochs = num_epochs
        self.epoch = 0 # current epoch
        self.batch_size = batch_size
        self.eval_on_train = eval_on_train # evaluation after each iteration
        self.metrics_ops = metrics_ops # metrics

    def train_and_eval(self, train_x, train_y, test_x=None, test_y=None):
        # init variables and weights
        self._variable_weights_init()
        # train
        for self.epoch in range(self.num_epochs):
            self.train(train_x, train_y, self.epoch)
            # validation
            if self.eval_on_train and test_x is not None and test_y is not None:
                self.eval(test_x, test_y)
    
    def train(self, train_x, train_y, epoch):
        iter_start_time = time.time()
        for i in range(len(list(train_x.values())[0])):
            # get input value
            input_values = {}
            for input_node_name in train_x.keys():
                input_values[input_node_name] = train_x[input_node_name][i]
            self.one_step(input_values, train_y[i], is_eval=False)
            # batch update
            if (i + 1) % self.batch_size == 0:
                # print("epoch: {:d}, {:.3f} sec/iter, iter: {:d}/{:d}, loss: {:.3f}".format(epoch + 1, time.time() - iter_start_time, i + 1, len(list(train_x.values())[0]), self.loss.value[0, 0]))
                iter_start_time = time.time()
                self._optimzer_update()
               
                

    def eval(self, test_x, test_y):
        for metrics_op in self.metrics_ops:
            metrics_op.reset_value()

        for i in range(len(list(test_x.values())[0])):
            input_values = {}
            for input_node_name in test_x.keys():
                input_values[input_node_name] = test_x[input_node_name][i]
            self.one_step(input_values, test_y[i], is_eval=True)

            for metrics_op in self.metrics_ops:
                metrics_op.forward()
        
        # print metrics
        metrics_msg = 'Epoch {:d}, Metrics: '.format(self.epoch + 1)
        for metrics_op in self.metrics_ops:
            metrics_msg += metrics_op.value_str()
        print(metrics_msg)

    def one_step(self, data_x, data_y, is_eval=False):
        # set input value
        for i in range(len(self.inputs)):
            input_value = data_x.get(self.inputs[i].name)
            self.inputs[i].set_value(np.mat(input_value).T)
        # set label value
        self.labels.set_value(np.mat(data_y).T)
        
        if not is_eval:
            self.optimzer.one_step()
    
    @abc.abstractmethod
    def _variable_weights_init(self):
        """
        Initiate variables and weights
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def _optimzer_update(self):
        """
        Update weights via optimizer.
        """
        raise NotImplementedError()

class SimpleTrainer(Trainer):
    def __init__(self, *args, **kargs):
        Trainer.__init__(self, *args, **kargs)

    def _variable_weights_init(self):
        pass

    def _optimzer_update(self):
        self.optimzer.update()

