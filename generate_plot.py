import numpy as np
import plotly.graph_objects as go

from data_reader import load_weights

w = load_weights()
w = np.delete(w, len(w) - 1)

heat = np.reshape(w, (28, 28)).tolist()

fig = go.Figure(data=go.Heatmap(z=heat, colorscale='greys'))
fig.update_layout(width=800, height=800)
fig.write_image('output/heatmap.png')
fig.show()
