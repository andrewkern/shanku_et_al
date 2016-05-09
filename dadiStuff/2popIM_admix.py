import dadi
import numpy
import scipy
import pyOpt
import dadiFunctions

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
except:
    raise ImportError('mpi4py is required for parallelization')


data = dadi.Spectrum.from_file('zi_na.auto.recomb.dadi_178_130.fs')

ns=[178,130]
pts_l = [1000]

# The Demographics1D and Demographics2D modules contain a few simple models,
# mostly as examples. We could use one of those.
func = dadiFunctions.IM_misorient_admix
#params = (nu1_0,nu1_1,nu1_2,nu1_3,nu1_4,nu2_0,nu2_1,nu2_2,nu2_3,nu2_4,t0,t1,t2,t3,t4,m12,m21,p_misid)



params = [  1.53508047e+00,   2.01954315e-01,   7.05672270e+00,
         1.04123880e+00,   2.43171832e-01,   6.95840502e-01,
         1.12649753e+00,   8.02383249e-04,   9.52992145e-06,
         2.92856419e-02]
upper_bound = [5,1,20, 10, 2, 10,10,0.15,0.5,0.25]
lower_bound = [1e-2,1e-2, 1e-2, 1e-2, 0, 0, 0,0,0,0,0,0]

p0 = params
#p0=dadiFunctions.makeRandomParams(lower_bound,upper_bound)
#p0=[5.43844318e+00,   8.98581374e-02,   9.10712839e+00,
#         5.90321729e-01,   8.83496283e-01,   5.50673559e-01,
#         8.97804733e-02,   4.25978610e-03,   3.21663638e-02]
#perturb these a bit
p1 = dadi.Misc.perturb_params(p0, fold=0.5, upper_bound=upper_bound)

func_ex = dadi.Numerics.make_extrap_func(func)

# Instantiate Optimization Problem 

#lamda definition of objective, stuffing constraint into return value
#objfunc = lambda x: (dadi.Inference._object_func(x, data, func_ex, pts_l, 
#                                   lower_bound=lower_bound,
#                                   upper_bound=upper_bound),(x[4]-x[3]),0)

def objfunc(x):
	f = dadi.Inference._object_func(x, data, func_ex, pts_l, 
	                                  lower_bound=lower_bound,
                                          upper_bound=upper_bound)
	g=[]
#	g = [0.0]*2
#	g[0] = x[3]-x[4]
#	g[1] = x[3]-x[2]
	fail = 0
	return f,g,fail
	
# (nu1_0,nu2_0,nu1,nu2,T,m12,m21,t_ad,p_ad,p_misid)
opt_prob = pyOpt.Optimization('dadi optimization',objfunc)
opt_prob.addVar('nu1_0','c',lower=lower_bound[0],upper=upper_bound[0],value=p1[0])
opt_prob.addVar('nu2_0','c',lower=lower_bound[1],upper=upper_bound[1],value=p1[1])
opt_prob.addVar('nu1','c',lower=lower_bound[2],upper=upper_bound[2],value=p1[2])
opt_prob.addVar('nu2','c',lower=lower_bound[3],upper=upper_bound[3],value=p1[3])
opt_prob.addVar('T','c',lower=lower_bound[4],upper=upper_bound[4],value=p1[4])
opt_prob.addVar('mAf_NA','c',lower=lower_bound[5],upper=upper_bound[5],value=p1[5])
opt_prob.addVar('mNA_Af','c',lower=lower_bound[6],upper=upper_bound[6],value=p1[6])
opt_prob.addVar('t_ad','c',lower=lower_bound[7],upper=upper_bound[7],value=p1[7])
opt_prob.addVar('p_ad','c',lower=lower_bound[8],upper=upper_bound[8],value=p1[8])
opt_prob.addVar('p_misid','c',lower=lower_bound[9],upper=upper_bound[9],value=p1[9])
opt_prob.addObj('f')
#opt_prob.addConGroup('g',2,'i')

if myrank == 0:
	print opt_prob


#optimize
psqp = pyOpt.ALPSO(pll_type='DPM')
psqp.setOption('printOuterIters',1)
#psqp.setOption('maxOuterIter',1)
#psqp.setOption('stopCriteria',0)
psqp.setOption('SwarmSize',64)
psqp(opt_prob)
print opt_prob.solution(0)

popt = numpy.zeros(len(p1))
for i in opt_prob._solutions[0]._variables:
    popt[i]= opt_prob._solutions[0]._variables[i].__dict__['value']

model = func_ex(popt, ns, pts_l)
ll_opt = dadi.Inference.ll_multinom(model, data)

if myrank == 0:
	print 'Optimized log-likelihood:', ll_opt
	print 'AIC:',(-2*ll_opt) + (2*(len(popt)))
	#scaled estimates
	theta0 = dadi.Inference.optimal_sfs_scaling(model, data)
	L = 19102507 + 14258355 + 17945602 + 19296650
	Nref= theta0 / 5.49e-9 / L / 4
	

	print 'Nref:',Nref
	paramsTxt =['nu1_0','nu2_0','nu1','nu2','T','2Nref_m12','2Nref_m21','t_ad','p_ad','p_misid']
	scaledParams = [Nref*popt[0],Nref*popt[1],Nref*popt[2],Nref*popt[3],2*Nref/15*popt[4],popt[5],
	                popt[6],2*Nref/15*popt[7],popt[8],popt[9]]
	for i in range(len(paramsTxt)):
		print paramsTxt[i],':',str(scaledParams[i])
	print ""
	print repr(popt)

############### 
# Now refine the optimization using Local Optimizer
# Instantiate Optimizer (SLSQP) 
# Instantiate Optimizer (SLSQP)
slsqp = pyOpt.SLSQP()
# Solve Problem (With Parallel Gradient)
if myrank == 0:
	print 'going for second optimization'

slsqp(opt_prob.solution(0),sens_type='FD',sens_mode='pgc')
print opt_prob.solution(0).solution(0)
opt = numpy.zeros(len(p1))
for i in opt_prob._solutions[0]._solutions[0]._variables:
	popt[i]= opt_prob._solutions[0]._solutions[0]._variables[i].__dict__['value']
	# 
model = func_ex(popt, ns, pts_l)
ll_opt = dadi.Inference.ll_multinom(model, data)
if myrank == 0:	  
	print 'After Second Optimization'
	print 'Optimized log-likelihood:', ll_opt
	print 'AIC:',(-2*ll_opt) + (2*(len(popt)-4))

	#scaled estimates
	theta0 = dadi.Inference.optimal_sfs_scaling(model, data)
	print 'with u = 5.49e-9'
	L = 19102507 + 14258355 + 17945602 + 19296650
	Nref= theta0 / 5.49e-9 / L / 4

	print 'Nref:',Nref
	paramsTxt =['nu1_0','nu2_0','nu1','nu2','T','2Nref_m12','2Nref_m21','t_ad','p_ad','p_misid']
	scaledParams = [Nref*popt[0],Nref*popt[1],Nref*popt[2],Nref*popt[3],2*Nref/15*popt[4],popt[5],
	                popt[6],2*Nref/15*popt[7],popt[8],popt[9]]
	for i in range(len(paramsTxt)):
		print paramsTxt[i],':',str(scaledParams[i])
	print ""
	print repr(popt)
