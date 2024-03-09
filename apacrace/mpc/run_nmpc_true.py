"""	Nonlinear MPC using dynamic bicycle model.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import time as tm
import numpy as np
import casadi
import _pickle as pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from bayes_race.params import ORCA
from bayes_race.models import Dynamic
from bayes_race.tracks import ETHZ
from bayes_race.mpc.planner import ConstantSpeed
from bayes_race.mpc.nmpc import setupNLP
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
        #   'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

#####################################################################
# CHANGE THIS

SAVE_RESULTS = False
TRACK_CONS = False

#####################################################################
# default settings

SAMPLING_TIME = 0.02
HORIZON = 20
COST_Q = np.diag([1, 1])
COST_P = np.diag([0, 0])
COST_R = np.diag([5/1000, 1])

if not TRACK_CONS:
	SUFFIX = 'NOCONS-'
else:
	SUFFIX = ''

#####################################################################
# load vehicle parameters

params = ORCA(control='pwm')
model = Dynamic(**params)

#####################################################################
# load track

TRACK_NAME = 'ETHZ'
track = ETHZ(reference='optimal', longer=True)
SIM_TIME = 34.6

#####################################################################
# extract data

Ts = SAMPLING_TIME
n_steps = int(SIM_TIME/Ts)
n_states = model.n_states
n_inputs = model.n_inputs
horizon = HORIZON

#####################################################################
# define controller

nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, model, track, track_cons=TRACK_CONS)

#####################################################################
# closed-loop simulation

# initialize
states = np.zeros([n_states, n_steps+1])
dstates = np.zeros([n_states, n_steps+1])
inputs = np.zeros([n_inputs, n_steps])
time = np.linspace(0, n_steps, n_steps+1)*Ts
Ffy = np.zeros([n_steps+1])
Frx = np.zeros([n_steps+1])
Fry = np.zeros([n_steps+1])
hstates = np.zeros([n_states,horizon+1])
hstates2 = np.zeros([n_states,horizon+1])

projidx = 0
x_init = np.zeros(n_states)
x_init[0], x_init[1] = track.x_init, track.y_init
x_init[2] = track.psi_init
x_init[3] = track.vx_init
dstates[0,0] = x_init[3]
states[:,0] = x_init
print('starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))

# dynamic plot
H = .2
W = .1
dims = np.array([[-H/2.,-W/2.],[-H/2.,W/2.],[H/2.,W/2.],[H/2.,-W/2.],[-H/2.,-W/2.]])

fig_track = track.plot(color='k', grid=False)
plt.plot(track.x_raceline, track.y_raceline, '--k', alpha=0.5, lw=0.5)
ax = plt.gca()
LnS, = ax.plot(states[0,0], states[1,0], 'r', label='Trajectory', alpha=0.8)
# LnR, = ax.plot(states[0,0], states[1,0], '-b', marker='o', markersize=1, lw=0.5, label="reference")
xyproj, _ = track.project(x=x_init[0], y=x_init[1], raceline=track.raceline)
LnP, = ax.plot(states[0,0] + dims[:,0]*np.cos(states[2,0]) - dims[:,1]*np.sin(states[2,0])\
		, states[1,0] + dims[:,0]*np.sin(states[2,0]) + dims[:,1]*np.cos(states[2,0]), 'purple', alpha=0.8, label='Current pose')
LnH, = ax.plot(hstates[0], hstates[1], '-g', marker='o', markersize=1, lw=0.5, color='green', label="ground truth")
LnH2, = ax.plot(hstates2[0], hstates2[1], '-r', marker='o', markersize=1, lw=0.5, color='blue', label="prediction")
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()

plt.figure()
plt.grid(True)
ax2 = plt.gca()
LnFfy, = ax2.plot(0, 0, label='Ffy')
LnFrx, = ax2.plot(0, 0, label='Frx')
LnFry, = ax2.plot(0, 0, label='Fry')
plt.xlim([0, SIM_TIME])
plt.ylim([-params['mass']*9.81, params['mass']*9.81])
plt.xlabel('time [s]')
plt.ylabel('force [N]')
plt.legend()
plt.ion()
plt.show()
ref_speeds = []
Drs = []
Dfs = []

# main simulation loop
for idt in range(n_steps-horizon):
		
	uprev = inputs[:,idt-1]
	x0 = states[:,idt]
	Drs.append(model.Dr)
	Dfs.append(model.Df)
	
	if idt > 720 :
		model.Df -= model.Df/2200.
		model.Dr -= model.Dr/2200.
		params['Dr'] -= params['Dr']/2200.
		params['Df'] -= params['Df']/2200.
	# planner based on BayesOpt
	if idt > 2 :
		xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx, curr_mu=(model.Df+model.Dr)/(9.81*params['mass']))
	else :
		xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx)
	ref_speeds.append(v)
	if projidx > 656 :
		projidx = 0
	# print(xref)
	start = tm.time()
	nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, model, track, track_cons=TRACK_CONS)
	umpc, fval, xmpc = nlp.solve(x0=x0, xref=xref[:2,:], uprev=uprev)
	# print(xmpc)
	end = tm.time()
	inputs[:,idt] = umpc[:,0]
	print("iter: {}, cost: {:.5f}, time: {:.2f}".format(idt, fval, end-start))
	
	# update current position with numerical integration (exact model)
	x_next, dxdt_next = model.sim_continuous(states[:,idt], inputs[:,idt].reshape(-1,1), [0, Ts])
	states[:,idt+1] = x_next[:,-1]
	dstates[:,idt+1] = dxdt_next[:,-1]
	Ffy[idt+1], Frx[idt+1], Fry[idt+1] = model.calc_forces(states[:,idt], inputs[:,idt])

	# forward sim to predict over the horizon
	hstates[:,0] = x0
	hstates2[:,0] = x0
	for idh in range(horizon):
		x_next, dxdt_next = model.sim_continuous(hstates[:,idh], umpc[:,idh].reshape(-1,1), [0, Ts])
		hstates[:,idh+1] = x_next[:,-1]
		hstates2[:,idh+1] = xmpc[:,idh+1]

	# update plot
	LnS.set_xdata(states[0,:idt+1])
	LnS.set_ydata(states[1,:idt+1])

	# LnR.set_xdata(xref[0,1:])
	# LnR.set_ydata(xref[1,1:])

	LnP.set_xdata(states[0,idt] + dims[:,0]*np.cos(states[2,idt]) - dims[:,1]*np.sin(states[2,idt]))
	LnP.set_ydata(states[1,idt] + dims[:,0]*np.sin(states[2,idt]) + dims[:,1]*np.cos(states[2,idt]))
	
	LnH.set_xdata(hstates[0])
	LnH.set_ydata(hstates[1])

	LnH2.set_xdata(hstates2[0])
	LnH2.set_ydata(hstates2[1])
	
	LnFfy.set_xdata(time[:idt+1])
	LnFfy.set_ydata(Ffy[:idt+1])

	LnFrx.set_xdata(time[:idt+1])
	LnFrx.set_ydata(Frx[:idt+1])

	LnFry.set_xdata(time[:idt+1])
	LnFry.set_ydata(Fry[:idt+1])

	plt.pause(Ts/100)

plt.ioff()

#####################################################################
# save data

if SAVE_RESULTS:
	np.savez(
		'../data/DYN-NMPC-{}{}.npz'.format(SUFFIX, TRACK_NAME),
		time=time,
		states=states,
		dstates=dstates,
		inputs=inputs,
		)

#####################################################################
# plots

# plot speed
plt.figure()
vel = np.sqrt(dstates[0,:]**2 + dstates[1,:]**2)
plt.plot(time[:n_steps-horizon], vel[:n_steps-horizon], label='abs')
plt.plot(time[:n_steps-horizon], states[3,:n_steps-horizon], label='vx')
plt.plot(time[:n_steps-horizon], states[4,:n_steps-horizon], label='vy')
plt.plot(time[:n_steps-horizon], ref_speeds, label='ref_speeds')
plt.xlabel('time [s]')
plt.ylabel('speed [m/s]')
plt.grid(True)
plt.legend()

# plot acceleration
plt.figure()
plt.plot(time[:n_steps-horizon], inputs[0,:n_steps-horizon])
plt.xlabel('time [s]')
plt.ylabel('PWM duty cycle [-]')
plt.grid(True)

# plot steering angle
plt.figure()
plt.plot(time[:n_steps-horizon], inputs[1,:n_steps-horizon])
plt.xlabel('time [s]')
plt.ylabel('steering [rad]')
plt.grid(True)

# plot inertial heading
plt.figure()
plt.plot(time[:n_steps-horizon], states[2,:n_steps-horizon])
plt.xlabel('time [s]')
plt.ylabel('orientation [rad]')
plt.grid(True)

plt.figure()
plt.plot(time[:n_steps-horizon], Dfs[:n_steps-horizon], label='Df')
plt.plot(time[:n_steps-horizon], Drs[:n_steps-horizon], label='Dr')
plt.xlabel('time [s]')
plt.ylabel('mu*N [N]')
plt.grid(True)
plt.legend()
plt.savefig('max_friction_forces_oracle.png')
fig_track.savefig('track_run.png')

plt.show()