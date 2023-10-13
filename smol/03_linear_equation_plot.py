import gradio as gr
import matplotlib.pyplot as plt
import numpy as np


def plot(w, b):
    x = np.linspace(-10, 10, 400)
    y = w * x + b
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axhline(0, color='black')  # Add horizontal line at y=0
    ax.axvline(0, color='black')  # Add vertical line at x=0
    ax.plot(x, y)
    ax.set_ylim(-10, 10)
    ax.set_xlim(-10, 10)
    ax.grid(True)
    ax.set_title('y = w * x + b')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return gr.Plot(fig)

iface = gr.Interface(fn=plot, inputs=[gr.components.Slider(0, 4), gr.components.Slider(0, 10)], outputs="plot", live=True)

iface.launch()
