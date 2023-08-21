
import pyomo.environ as pe
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



def bwm_opt_fn(n, best, worst, b_comparisons, w_comparisons):
    M = pe.ConcreteModel() # instancia modelo
    
    # Sets
    M.criteria_set = pe.Set(initialize = range(n))  #cria set de critérios no pyomo
    
    # Variables
    weights = M.weights = pe.Var(M.criteria_set, within = pe.NonNegativeReals)
    e = M.e = pe.Var(within = pe.NonNegativeReals)
    
    #Parameters
    M.best = pe.Param(initialize = best)
    M.worst = pe.Param(initialize = worst)
    M.b_comparisons = pe.Param(M.criteria_set, initialize = b_comparisons)
    M.w_comparisons = pe.Param(M.criteria_set, initialize = w_comparisons)
    
    #Objective
    M.obj = pe.Objective(rule = e, sense = pe.minimize)
    
    #Constraints
    M.Cons_1 = pe.ConstraintList()
    for j in M.criteria_set:
        M.Cons_1.add(weights[M.best] - M.b_comparisons[j] * weights[j] <= e)
        M.Cons_1.add(-weights[M.best] + M.b_comparisons[j] * weights[j] <= e)
    
    M.Cons_2 = pe.ConstraintList()
    for j in M.criteria_set:
        M.Cons_2.add(weights[j] - M.w_comparisons[j] * weights[M.worst] <= e)
        M.Cons_2.add(-weights[j] + M.w_comparisons[j] * weights[M.worst] <= e)
    
    M.Cons_3 = pe.Constraint(rule = lambda M: sum(M.weights[i] for i in M.criteria_set) == 1)  
    return M 



def Step (b_comparisons, w_comparisons, data, weights_prof_micro):
    
    # inputs for the bwm function
    n_macro = 5
    best = 0
    worst = 1
    
    # run bwm to get the weights of the criteria (macro) 
    M = bwm_opt_fn(n_macro, best, worst, b_comparisons, w_comparisons)
    pe.SolverFactory('cplex').solve(M)
    
    # get the weights from the variables  
    var_list = [pe.value(v) for v in  M.component_data_objects([pe.Var])]
    weight_prof_macro = var_list[0]
    weights_cost_criteria = np.array(var_list[1:-1])
    
    # Now calculate get the final weight vector considering the 3 profitability criteria 
    weights_prof = weights_prof_micro * weight_prof_macro
    
    # Concatenate the profitabilty weights with the weights that came from the other criteria
    weights = np.concatenate([weights_prof, weights_cost_criteria])

    # pandas DataFrame to numpy array
    X = np.array(data)
    # Apply MOORA normalization 
    X_norm = X/np.linalg.norm(X, axis=0)
    # Multiply by the weights
    X_wnorm = X_norm * weights

    # apply MOORA
    criteria_type = ["max", "max", "max", "min", "min", "min", "min"]
    m = X_wnorm.shape[0]
    n = X_wnorm.shape[1]

    values = np.zeros(m)
    for i in range (m):
        for j in range (n):
            if criteria_type[j] == "min":
                values[i] -= X_wnorm[i,j] 
            else:
                values[i] += X_wnorm[i,j]

    # Apply the final Normalization
    values = values.reshape(-1,1)
    final_result = MinMaxScaler().fit(values).transform(values)
    
    # Get a table of results
    table = pd.DataFrame(final_result, columns=["Value"], index=data.index)
    table["Position"] = table['Value'].rank(ascending=False)
    
    return table