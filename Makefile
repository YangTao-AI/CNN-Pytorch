all:
	cat Makefile

clean:
	rm -rf train_log/

clean-trash:
	./scripts/clean_trash.sh

