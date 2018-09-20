import json, re, csv
from IPython import embed

path = './train_log/resnet18,pretrained,lr:0.01,wd:0.0005:Al/classes.json'
with open(path, 'r') as f:
    classes = json.load(f)


with open('result.txt') as f:
    content = f.readlines()
mp = {
    '正常':         'norm',
    '不导电':       'defect1',
    '擦花':         'defect2',
    '横条压凹':     'defect3',
    '桔皮':         'defect4',
    '漏底':         'defect5',
    '碰伤':         'defect6',
    '起坑':         'defect7',
    '凸粉':         'defect8',
    '涂层开裂':     'defect9',
    '脏点':         'defect10',
    '其他':         'defect11',
}
pat = re.compile('.*/([0-9]*)\.jpg ([0-9]*).*?', re.S)

out = open('result.csv', 'w')
writer = csv.writer(out, dialect='excel')
t = []
for each in content:
    data = pat.findall(each)
    x = int(data[0][0])
    y = int(data[0][1])
    z = [x, mp[classes[y]]]
    t.append(z)

t.sort(key = lambda x: x[0])
for z in t:
    writer.writerow(z)
    
