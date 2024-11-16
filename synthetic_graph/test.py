import numpy as np
from graph_cython_numpy import Input, Sin, Integrator


bins = Input(data=np.linspace(0, np.pi, 100).tolist())

values = Sin()
bins >> values

integrator = Integrator()
values >> integrator
bins >> integrator

integrator.compile()
result = integrator.run()
print(result[0])
