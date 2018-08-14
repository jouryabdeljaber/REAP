import numpy as np
from pyomo.environ import *
from pyomo.mpec import *

# Create a solver
opt = SolverFactory('path',executable='./pathampl')#)'/home/mathew/Downloads/ampl.linux64/ampl')

#
# A simple model with binary variables and
# an empty constraint list.
#
'''
model = pyo.AbstractModel()
model.n = pyo.Param(default=4)
model.x = pyo.Var(pyo.RangeSet(model.n), within=pyo.Binary)
def o_rule(model):
    return pyo.summation(model.x)
model.o = pyo.Objective(rule=o_rule)
model.c = pyo.ConstraintList()

# Create a model instance and optimize
instance = model.create_instance()
results = opt.solve(instance)
instance.display()
'''
def addLCVectorConstraint(model, M, q, vars, I):
    lc_rule = lambda m, i: \
        complements(sum(M[i-1,j-1] * vars[j] for j in I) + q[i-1] >= 0, vars[i] >= 0)
    model.c = Complementarity(I, rule=lc_rule)




def solveLCP(M,q):
    model = ConcreteModel()
    #model.x1 = Var()  
    #model.f1 = Complementarity(expr=
    #complements(model.x1 >= 0,
    #squ(model.x1-1) >= 0))
    DIM = q.size

    model.I = RangeSet(1,DIM)
    model.z = Var(model.I, domain=NonNegativeReals)
    addLCVectorConstraint(model, M, q, model.z, model.I)
    #for i in range(0,DIM):
    #    model.z[i+1] = 0
    results = opt.solve(model)#, warmstart=True)
    ret = np.zeros(DIM)
    for i in range(0,DIM):
        ret[i] = value(model.z[i+1])
    return ret