.PHONY: evaluate trans-driver all clean judge-% judge-all
CC = gcc
CLFAGS = -g
all: trans-driver evaluate

judge-%:
	@python3 judge.py 'input/$*_0.in'

judge-all:
	@python3 judge.py 'input/*_*.in'

evaluate: trans-driver
	@$(CC) $(CLFAGS) -o evaluate evaluate.c

trans-driver:
	@$(CC) -DMAX_N=32 $(CLFAGS) -o trans-driver32 trans-driver.c trans.c
	@$(CC) -DMAX_N=64 $(CLFAGS) -o trans-driver64 trans-driver.c trans.c 

clean:
	@rm -f .csim_results .marker
	@rm -f trans-driver32 trans-driver64 evaluate
	@rm -f *full.trace *trans32.trace *trans64.trace *hitf32 *hitf64
	@rm -f answer/* output/* logs/*