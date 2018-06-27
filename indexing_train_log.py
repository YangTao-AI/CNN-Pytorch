import os, json, csv

paths = ['train_log']

argss = []
keys = set()


for path in paths:
    targets = os.listdir(path)
    for target in targets:
        current_path = os.path.join(path, target)
        args_path = os.path.join(current_path, 'args.txt')
        if not os.path.isfile(args_path):
            continue

        with open(args_path, 'r') as f:
            args = json.load(f)

        args['__path__'] = os.path.abspath(current_path)
        argss.append(args)
        keys.update(set(args.keys()))

keys = sorted(list(keys))


with open('train_log.csv','w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(keys)

    for args in argss:
        data = []
        for key in keys:
            value = args[key] if key in args else None
            data.append(value)
        writer.writerow(data)

