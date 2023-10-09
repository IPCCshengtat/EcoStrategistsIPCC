import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real
import math
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination

def runopt(elec_requirement, elec_nat, elec_POME, elec_solar, cost_nat, cost_POME, cost_solar, 
           co2_nat, co2_pome, co2_solar, solar_land_max, solar_panel_each, POME_max, POME_each, 
           w_cost_nat, w_cost_POME, w_cost_solar, w_co2_nat, w_co2_POME, w_co2_solar):
    solar_no_max = math.floor(solar_land_max/solar_panel_each)
    POME_no_max = math.floor(POME_max/POME_each)
    print("Something is written here")
    class PSEProblem(ElementwiseProblem):
        def __init__(self):
            vars = {
                "x1": Real(bounds = (0, 8*elec_nat)), 
                "x2": Real(bounds = (2*elec_POME, POME_no_max*elec_POME)),
                "x3": Real(bounds = (0, solar_no_max*elec_solar))
            }
            super().__init__(n_vars = 3,
                            vars = vars, 
                            n_obj=2, 
                            n_ieq_constr = 1, 
                            xl = np.array([0, 2*elec_POME, 0]), 
                            xu = np.array([8*elec_nat, POME_no_max*elec_POME, solar_no_max*elec_solar]))

        def _evaluate(self, x, out, *args, **kwargs):
            x1, x2, x3 = x["x1"], x["x2"], x["x3"]
            # *(x1*1.35 - x1)////*abs(x2*0.9 - x2)////*abs(x3*0.85 - x3)
            # *(x1*1.1 - x1)////*abs(x2*8 - x2)////(x3*1.1 - x3)
            f1 = (cost_nat*x1*(x1*w_cost_nat - x1) + cost_POME*x2*abs(x2*w_cost_POME - x2) + cost_solar*x3*abs(x3*w_cost_solar - x3))/cost_solar
            f2 = (co2_nat*x1*(x1*w_co2_nat - x1) + co2_pome*x2*abs(x2*w_co2_POME - x2) + co2_solar*x3*(x3*w_co2_solar - x3))/co2_nat
            # f3 = np.ceil(x1/elec_nat)*elec_nat + np.ceil(x2/elec_POME)*elec_POME + np.ceil(x3/elec_solar)*elec_solar

            # normalization of the constraints inequality
            g1 = (elec_requirement - x1 - x2 - x3)/elec_requirement

            out["F"] = [f1, f2]
            out["G"] = [g1]
    print("Something is written here")
    problem = PSEProblem()

    algorithm = NSGA2(pop_size=100, # 300
                sampling=MixedVariableSampling(),
                n_offsprings=100, # 250
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(eta=30),
                mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                eliminate_duplicates=MixedVariableDuplicateElimination(),
                survival=RankAndCrowdingSurvival()
                )

    termination = get_termination("n_gen", 80) # 150
    print("Something is written here")

    res = minimize(problem, algorithm, termination, seed=5, save_history=True, verbose=True)
    columnname = ["f1_pymoo", "f2_pymoo"]
    fpandas = pd.DataFrame(res.F.tolist(), columns=columnname)
    resultdf = pd.DataFrame(res.X.tolist()).merge(fpandas,left_index=True, right_index=True)
    resultdf["x1 (int)"] = np.ceil(resultdf["x1"]/elec_nat)
    resultdf["x2 (int)"] = np.ceil(resultdf["x2"]/elec_POME)
    resultdf["x3 (int)"] = np.ceil(resultdf["x3"]/elec_solar)
    resultdf["f1_pymoo"] = (cost_nat*resultdf["x1"] + cost_POME*resultdf["x2"] + cost_solar*resultdf["x3"])
    resultdf["f2_pymoo"] = (co2_nat*resultdf["x1"] + co2_pome*resultdf["x2"] + co2_solar*resultdf["x3"])    
    resultdf["f1_calc_int"] = (cost_nat*resultdf["x1 (int)"]*elec_nat + cost_POME*resultdf["x2 (int)"]*elec_POME + cost_solar*resultdf["x3 (int)"]*elec_solar)
    resultdf["f2_calc_int"] = (co2_nat*resultdf["x1 (int)"]*elec_nat + co2_pome*resultdf["x2 (int)"]*elec_POME + co2_solar*resultdf["x3 (int)"]*elec_solar)
    
    index_reject_list = []
    index_reject_list_cont = []

    for index in range(len(resultdf)):
        compare_f1 = resultdf.loc[index, "f1_calc_int"]
        compare_f2 = resultdf.loc[index, "f2_calc_int"]
        for jedex in range(len(resultdf)):
            if (compare_f1 < resultdf.loc[jedex, "f1_calc_int"]) and (compare_f2 < resultdf.loc[jedex, "f2_calc_int"]):
                index_reject_list.append(jedex)
    
    for index in range(len(resultdf)):
        compare_f1 = resultdf.loc[index, "f1_pymoo"]
        compare_f2 = resultdf.loc[index, "f2_pymoo"]
        for jedex in range(len(resultdf)):
            if (compare_f1 < resultdf.loc[jedex, "f1_pymoo"]) and (compare_f2 < resultdf.loc[jedex, "f2_pymoo"]):
                index_reject_list_cont.append(jedex)
            
    index_reject_list = pd.Series(index_reject_list).unique().tolist()
    index_reject_list_cont = pd.Series(index_reject_list_cont).unique().tolist()

    resultdf['Dominance'] = 'Dominant'
    resultdf.loc[index_reject_list, "Dominance"] = "Non-Dominant"

    resultdf['Dominance_Cont'] = 'Dominant'
    resultdf.loc[index_reject_list_cont, "Dominance_Cont"] = "Non-Dominant"
    
    resultdf["f1_pymoo_norm"] = resultdf["f1_pymoo"]/resultdf["f1_pymoo"].max()
    resultdf["f2_pymoo_norm"] = resultdf["f2_pymoo"]/resultdf["f2_pymoo"].max()

    resultdf["f1_cal_int_norm"] = resultdf["f1_calc_int"]/resultdf["f1_calc_int"].max()
    resultdf["f2_cal_int_norm"] = resultdf["f2_calc_int"]/resultdf["f2_calc_int"].max()

    return resultdf
