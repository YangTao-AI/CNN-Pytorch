from sklearn.metrics import f1_score
import numpy as np
import pickle

with open('val.pkl', 'rb') as f:
    data = pickle.load(f)


eps = 1e-12

def calc(y_true, y_pred):
    return f1.mean()



def f1(y_true, y_pred):
    tp = (y_true == y_pred).sum(axis=0)
    fp = ((1-y_true) == y_pred).sum(axis=0)
    fn = (y_true == (1-y_pred)).sum(axis=0)
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2*p*r / (p+r+eps)
    return f1.mean()

y_true = []
y_pred = []
for out, gt in data:
    out = 1 / (1 + np.exp(-out))
    y_pred.append(out)
    y_true.append(gt)


_y_true = np.concatenate(y_true, axis=0)
_y_pred = np.concatenate(y_pred, axis=0)


def calc(th):
    y_pred = _y_pred.copy()
    y_true = _y_true.copy()
    y_pred[y_pred>th] = 1
    y_pred[y_pred<1] = 0
    return f1_score(y_true, y_pred, average='macro')
    

d = [0.2] * 28
print(calc(d))
ans = 0
for i in range(0, 28):
    pp = 0
    j = 0
    good = 0.5
    while j < 1:
        d[i] = j
        tmp = calc(d)
        if tmp >= pp:
            pp = tmp
            good = j
            print(i, round(j*10), tmp, d)
        j += 0.05
    d[i] = good

print(d, calc(d))
