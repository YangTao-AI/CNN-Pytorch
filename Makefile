all:
	cat Makefile

clean:
	rm -rf train_log/

clean-trash:
	./scripts/clean_trash.sh

res18:
	python3 main.py -a resnet18 --pretrained --epochs 100 -b 256 --lr 1e-2 --wd 1e-2  --data-cached -j 8

res50:
	python3 main.py -a resnet50 --pretrained --epochs 100 -b 64 --lr 1e-2 --wd 1e-2  --data-cached -j 8

