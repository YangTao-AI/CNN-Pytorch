python3 train_nori.py --epochs 200 --logdir resnet50-imagenet -a resnet50 --lr=0.1 --weight-decay=5e-4 --batch-size=256 -j 8
python3 train_nori.py --epochs 200 --logdir resnet18-imagenet -a resnet18 --lr=0.1 --weight-decay=5e-4 --batch-size=256 -j 8

