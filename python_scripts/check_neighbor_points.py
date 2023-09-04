import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from glob import glob
# Create figure
fig = go.Figure()

file_list = glob("output/neighbor_list_*.txt")
full_data = np.loadtxt("output/boundary_points.txt")
full_data = np.concatenate((full_data, np.zeros((full_data.shape[0], 1))), axis=1)
print(full_data.shape)
# Add traces, one for each slider step
for i in range(50):
    data = np.loadtxt(file_list[i])
    if len(data.shape) == 1:
        data = data.reshape((1, -1))
    data = np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)
    data[-1, -1] = 0.5
    data = np.concatenate((full_data, data), axis=0)
    fig.add_trace(
        go.Scatter3d(
            visible=False,
            x=data[:, 0], y=data[:, 1], z=data[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=data[:, 3],  # set color to an array/list of desired values
                colorscale='Viridis',  # choose a colorscale
                opacity=0.8

            )
        )
    )

# Make 10th trace visible
fig.data[0].visible = True

# Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

fig.show()