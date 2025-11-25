import gurobipy as gp
import numpy as np
from gurobipy import GRB


def convert_to_slack_form(model):
    slack_model = gp.Model("Slack_Form")
    slack_model.setParam('OutputFlag', 0)

    # Copy existing decision variables, making all variables continuous
    var_map = {v.varName: slack_model.addVar(lb=v.lb, vtype='C', name=v.varName)
               for v in model.getVars()}

    slack_model.update()  # Ensure new variables are added

    for i, constr in enumerate(model.getConstrs()):
        sense = constr.sense
        lhs = gp.LinExpr()
        row = model.getRow(constr)
        for j in range(row.size()):
            lhs += row.getCoeff(j) * var_map[row.getVar(j).varName]
        rhs = constr.rhs

        if sense == GRB.LESS_EQUAL:  # ≤ constraint → add slack
            slack_var = slack_model.addVar(lb=0, name=f"slack_{i}")
            slack_model.addConstr(lhs + slack_var == rhs)

        elif sense == GRB.GREATER_EQUAL:  # ≥ constraint → multiply by -1, then add slack
            slack_var = slack_model.addVar(lb=0, name=f"slack_{i}")
            slack_model.addConstr(-lhs + slack_var == -rhs)

        else:  # Equality constraints remain unchanged
            slack_model.addConstr(lhs == rhs)

    # Add explicit upper bound constraints for the original variables
    for var in model.getVars():
        if var.ub != float('inf'):  # Only for variables with upper bounds
            slack_var = slack_model.addVar(lb=0, name=f"slack_{var.varName}_ub")
            slack_model.addConstr(var_map[var.varName] + slack_var == var.ub)

    # Copy the objective function
    obj_expr = gp.LinExpr()
    for var in model.getVars():
        obj_expr += var.obj * var_map[var.varName]

    slack_model.setObjective(obj_expr, model.ModelSense)  # Minimize or maximize

    slack_model.update()
    return slack_model  # Return the new slack form model


def getConstraintsMatrixFormSlackModel(model):
    xs = model.getVars()
    A = []
    b = []
    for constr in model.getConstrs():
        if constr.sense != GRB.EQUAL:
            raise Exception("Constraints have to be of the form Ax == b")
        a_i = []
        for x in xs:
            a_i.append(model.getCoeff(constr, x))
        b_i = constr.rhs
        A.append(a_i)
        b.append(b_i)

    return np.array(A), np.array(b)
