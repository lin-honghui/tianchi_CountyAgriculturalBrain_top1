'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-11-28 06:02:00
'''
from ignite.metrics import Metric

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

class Label_Accuracy_drop_edge(Metric):
    """
    Calculates the accuracy.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...)
    - `y` must be in the following shape (batch_size, ...)
    """

    def __init__(self, n_class=5,edge_size=64):
        super(Label_Accuracy_drop_edge, self).__init__()
        self.n_class = n_class
        self.edge_size = edge_size

    def reset(self):
        self.step = 0
        self.mean_iu = 0

    def update(self, output):
        label_preds, label_trues = output
        # b c h w
        _,_,h,w = label_preds.shape
        label_preds = label_preds[:,:,self.edge_size:h-self.edge_size,self.edge_size:w-self.edge_size]
        label_trues = label_trues[:,:,self.edge_size:h-self.edge_size,self.edge_size:w-self.edge_size]
        label_preds = label_preds.max(dim=1)[1].data.cpu().numpy()
        label_preds = [i for i in label_preds]

        label_trues = label_trues.data.cpu().numpy()
        label_trues = [i for i in label_trues]

        hist = np.zeros((self.n_class, self.n_class))
        for lt, lp in zip(label_trues, label_preds):
            hist += _fast_hist(lt.flatten(), lp.flatten(), self.n_class)
        with np.errstate(divide='ignore', invalid='ignore'):
            iu = np.diag(hist) / (
                    hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
            )
        mean_iu = np.nanmean(iu)
        self.mean_iu += mean_iu
        self.step += 1

    def compute(self):
        return self.mean_iu / self.step
        
class Label_Accuracy(Metric):
    """
    Calculates the accuracy.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...)
    - `y` must be in the following shape (batch_size, ...)
    """

    def __init__(self, n_class=5):
        super(Label_Accuracy, self).__init__()
        self.n_class = n_class

    def reset(self):
        self.step = 0
        self.mean_iu = 0

    def update(self, output):
        label_preds, label_trues = output
        label_preds = label_preds.max(dim=1)[1].data.cpu().numpy()
        label_preds = [i for i in label_preds]

        label_trues = label_trues.data.cpu().numpy()
        label_trues = [i for i in label_trues]

        hist = np.zeros((self.n_class, self.n_class))
        for lt, lp in zip(label_trues, label_preds):
            hist += _fast_hist(lt.flatten(), lp.flatten(), self.n_class)
        with np.errstate(divide='ignore', invalid='ignore'):
            iu = np.diag(hist) / (
                    hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
            )
        mean_iu = np.nanmean(iu)
        self.mean_iu += mean_iu
        self.step += 1

    def compute(self):
        return self.mean_iu / self.step


