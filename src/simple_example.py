import gurobipy as gp
from gurobipy import GRB


# Create a new model
model = gp.Model("mip1")

# Create variables
x = model.addVar(vtype=GRB.BINARY, name="x")
y = model.addVar(vtype=GRB.BINARY, name="y")
z = model.addVar(vtype=GRB.BINARY, name="z")

# Set objective
model.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

# Add constraint: x + 2 y + 3 z <= 4
model.addConstr(x + 2 * y + 3 * z <= 4, "c0")

# Add constraint: x + y >= 1
model.addConstr(x + y >= 1, "c1")

# Optimize model
model.optimize()

for v in model.getVars():
    print('%s %g' % (v.VarName, v.X))

print('Obj: %g' % model.ObjVal)