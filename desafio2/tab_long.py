import numpy as np, sys
from agents import *
seed=int(sys.argv[1]); a=TabularQ(seed); r,c=train(a,200000,seed,eval_every=5000,eps_frac=0.5)
np.save('results/tabular_long_seed%d_curve.npy'%seed,c); print(seed,' '.join('%d:%.0f/%.0f'%(e/1000,m,100*l) for e,m,l in c[::4]), a.visited())
