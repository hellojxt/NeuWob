import plotly.express as px
import numpy as np

data = np.loadtxt("output/boundary_points.txt")
fig = px.scatter_3d(x=data[:, 0], y=data[:, 1], z=data[:, 2])
fig.show()