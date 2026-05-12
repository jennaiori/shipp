#%%

import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt


from shipp.kernel_pyomo import solve_lp_pyomo
from shipp.components import Storage, Production, TimeSeries
from shipp.kernel import solve_lp_sparse, build_lp_obj_revenues, build_lp_cst_sparse

from scipy.optimize import linprog, minimize, LinearConstraint


# --------
# Define input data
# --------

# Global input data for the numerical experiments
n_max = 8760  # number of time steps 
dt = 1 # time step duration [hour]
percent_bl = 0.99 # reliability of the baseload constraint
discount_rate = 0.03 # discount rate
n_year = 20  # Project duration [years]
p_min = 0 # Required minimum baseload power
p_cost_res = 3000  # USD/MW # cost per installed capacity of the renewable energy source [USD/MW]

# Input data for the storage characteristics
p_cost = 150*1e3  # cost per power capacity for the short term storage [USD/MW]
e_cost = 75 * 1e3 # cost per energy capacity for the short term storage [USD/MWh]
eta = 0.85 #Round trip efficiency for the short term storage

pyo_solver = 'none' # Name of the optimization solver (e.g. cplex, gurobi, mosek) - if 'none', default to linprog

# Input parameters to generate a dummy power and price signals
frequency_power = 0.5 
frequency_price = 5
mean_power = 50
price_low = 40
price_high = 60
p_max = 100

time = np.arange(0, n_max)/24

# The power is represented by a sine function of time
power = mean_power*(1.0+np.sin(time * 2*np.pi * frequency_power))

# The price is represented by the combinaison of a sine function of time and a random variation
price = np.random.uniform(price_low, price_high, n_max) + \
        10*np.sin(np.arange(0, n_max)/24 * 2*np.pi * frequency_price)

# Building objects representing the storage system
stor = Storage(e_cap = 100, p_cap = 50, eff_in = 1, eff_out= eta, e_cost = e_cost, p_cost = p_cost)
stor_null = Storage(e_cap = 0, p_cap = 0, eff_in =1, eff_out=1,  e_cost = 0, p_cost = 0) # Creation of a null object - necessary for the optimization function


#%% --------
# Create elements of optimization problem 
# --------

n = 5*24

options = dict(formulation = 'lp_alt', fixed_cap = True)

vec_obj = build_lp_obj_revenues(price[:n], n, options)

mat_eq, vec_eq, mat_ineq, vec_ineq, bounds_lower, bounds_upper =  build_lp_cst_sparse(power, dt, 0, p_max, n, stor, stor_null, stor1_p_cap_max = stor.p_cap, stor2_p_cap_max = stor_null.p_cap, stor1_e_cap_max = stor.e_cap, stor2_e_cap_max= stor_null.e_cap, options=options)


#%% --------
# Solve linear optimization problem with linprog
# --------

bounds = []
n_var = bounds_upper.shape[0]
for x in range(0, n_var):
    bounds.append((bounds_lower[x], bounds_upper[x]))

res_lp_linprog = linprog(vec_obj, A_ub= mat_ineq.toarray(), b_ub = vec_ineq, A_eq=mat_eq.toarray(), b_eq=vec_eq, bounds=bounds, method = 'highs-ipm')

print('LP objective function:\t', res_lp_linprog.fun)

plt.plot(res_lp_linprog.x[:n], '-.', label = 'LP linprog')
plt.legend()
plt.ylabel('Storage power [MW]')


#%% --------
# Solve non-linear optimization problem (including degradation) with minimize
# --------

alpha = 0.2 # Weight between revenue and degradation part of the objective function
e_target = 0.5*stor.e_cap

# Example of non-linear function for the degradation # TO BE REPLACED
def dummy_degradation(x):
    e = x[3*n:4*n+1]
    return np.linalg.norm(e - e_target)

# Example of gradient for the non-linear function for the degradation # TO BE REPLACED
def dummy_degradation_grad(x):
    e = x[3*n:4*n+1]
    v = e - e_target
    norm_v = np.linalg.norm(v)
    grad = np.zeros_like(x)
    if norm_v > 0:
        grad[3*n:4*n+1] = v / (norm_v)

    return grad

# Define constraints as with LinearConstraint objects - for algorithm trust-constr
constraints = [
    LinearConstraint(mat_ineq.toarray(), ub=vec_ineq),  # A_ub @ x <= b_ub
    LinearConstraint(mat_eq.toarray(), lb=vec_eq, ub=vec_eq)  # A_eq @ x == b_eq
]

x0 = res_lp_linprog.x

# Normalization factors for the objective function
norm_revenues = np.abs(np.dot(vec_obj, x0))
norm_degradation = dummy_degradation(x0)

# Objective function: 
# expressed as 
# f(x) = (1- alpha) * (-revenues)  + alpha * degradation
# Varying alpha allows to adjust the relative weight of revenues vs. degradation
def objective_with_degradation(x):

    return (1-alpha)*np.dot(vec_obj, x)/norm_revenues + alpha * dummy_degradation(x)/norm_degradation

# Gradient of the objective: c + degradation_grad
def objective_with_degradation_grad(x):
    return (1-alpha)*vec_obj/norm_revenues + alpha * dummy_degradation_grad(x)/norm_degradation

# Initial guess - start with the solution of the LP program since it satisfies the constraints of the problem
x0 = res_lp_linprog.x

# Solve using non-linear algorithm with gradients and Jacobians
res_nlp_minimize = minimize(
    objective_with_degradation,
    x0,
    jac = objective_with_degradation_grad,
    constraints=constraints,
    bounds=bounds,
    method='trust-constr',
    tol = 1e-5,
    options = dict(sparse_jacobian = True,
                    initial_tr_radius = 0.1,
                    verbose = 2 )
)


#%%

x_nlp = res_nlp_minimize.x

print('NLP execution time:\t{:.1f}s'.format(res_nlp_minimize.execution_time))

print('\t\t\tInitial point (x0)\tOptimum (x_nlp)')
print('Objective function\t{:.4f}\t\t\t{:.4f}'.format(objective_with_degradation(x0), objective_with_degradation(x_nlp)))
print('Storage revenues\t{:.2f}\t\t\t{:.2f}'.format(1e-3*np.dot(price[:n], x0[:n] ), 1e-3*np.dot(price[:n], x_nlp[:n] )))
print('Degradation\t\t{:.2f}\t\t\t{:.2f}'.format(dummy_degradation(x0),  dummy_degradation(x_nlp)))

#%%
plt.plot(res_lp_linprog.x[3*n:4*n+1], '-.', label = 'LP linprog')
plt.plot(res_nlp_minimize.x[3*n:4*n+1], ':', label = 'NLP minimize')
plt.ylabel('State of charge [MW]')
plt.legend()
plt.xlim([0, 48])
# %%
plt.plot(res_lp_linprog.x[:n], '-.', label = 'LP linprog')
plt.plot(res_nlp_minimize.x[:n], ':', label = 'NLP minimize')
plt.ylabel('Storage power [MW]')
plt.legend()
plt.xlim([0, 48])