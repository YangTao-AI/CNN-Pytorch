import pickle
import numpy as np
import datetime

_name = 'result_1542342712.8978148.pkl'
_name = 'result_1542365398.5426714.pkl'
_name = 'result_1542604563.1172318.pkl'

with open(_name, 'rb') as f:
    result = pickle.load(f)

with open('./data/sample_submission.csv', 'r') as f:
    c = f.readlines()

thre = np.array([0.44999999999999996, 0.7000000000000001, 0.6, 0.25, 0.25, 0.49999999999999994, 0.49999999999999994, 0.6, 0.7500000000000001, 0.44999999999999996, 0.3, 0.25, 0.39999999999999997, 0.25, 0.2, 0.44999999999999996, 0.2, 0.35, 0.2, 0.39999999999999997, 0.25, 0.3, 0.15000000000000002, 0.5499999999999999, 0.3, 0.39999999999999997, 0.49999999999999994, 0.5])
# thre = 0.5
f = open('ans/ans_{}_{}.csv'.format(_name, thre), 'w')


mp = {}
f.write('Id,Predicted\n')
for name, score in result:
    score = 1 / (1 + np.exp(-score))
    
    labels = list(np.where(score>thre)[0])
    if len(labels) == 0:
        # labels.append(np.argmax(score))
        # print(np.sort(score)[::-1])
        sss = np.argsort(score)[::-1]
        for i in range(0, 28):
            if (score[sss[0]] - score[sss[i]]) / score[sss[0]] < 0.1:
                labels.append(sss[i])
        # print(score[sss[0]], score[sss[1]], score[sss[2]])
        # print(labels)
        # input()


    labels.sort()
    labels = [str(label) for label in labels]
    
    mp[name] = '{},{}\n'.format(name, ' '.join(labels))

    # input()

for row in c:
    if row == 'Id,Predicted\n':
        continue
    name = row.split(',')[0]
    print(name, mp[name])
    f.write(mp[name])
