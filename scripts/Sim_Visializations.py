import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px



def plot_results_3d(t, voltages, save_path=None, interactive=False):
    # Plotly interactive 3D
    if interactive:
        neuron_ids = list(voltages.keys())
        cmap = px.colors.sequential.Viridis
        n_colors = len(cmap)

        fig = go.Figure()
        for i, neuron_id in enumerate(neuron_ids):
            v = voltages[neuron_id]
            color = cmap[int(i / max(1, len(neuron_ids)-1) * (n_colors-1))]
            fig.add_trace(go.Scatter3d(
                x=t,
                y=[i]*len(t),
                z=v,
                mode='lines',
                name=str(neuron_id),
                line=dict(color=color, width=4)
            ))

        fig.update_layout(
            title='Network Activity (3D View)',
            scene=dict(
                xaxis_title='Time (ms)',
                yaxis_title='Neuron ID',
                zaxis_title='Voltage (mV)',
            ),
            template='plotly_dark',
            width=1000,
            height=700,
            showlegend=True
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if save_path.endswith(".html"):
                fig.write_html(save_path)
            elif save_path.endswith(".png"):
                fig.write_image(save_path)
            else:
                raise ValueError("Unknown format for interactive 3D plot — use .html or .png")
        else:
            fig.show()
        return

    # Matplotlib static 3D
    else:
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection='3d')

        cmap = cm.get_cmap('viridis', len(voltages))
        for i, (neuron_id, v) in enumerate(voltages.items()):
            color = cmap(i)
            y = [i] * len(t)
            ax.plot(t, y, v, color=color, linewidth=2, label=str(neuron_id))

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron ID')
        ax.set_zlabel('Voltage (mV)')
        ax.set_title('Network Activity (3D view)')
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
        return



def plot_results(t, voltages, stim_times=[], save_path=None, interactive=False):
    # Plotly interactive 2D
    if interactive:
        colors = px.colors.sequential.Viridis
        fig = go.Figure()

        neuron_ids = list(voltages.keys())
        n_colors = len(colors)

        for i, (neuron_id, v) in enumerate(voltages.items()):
            color = colors[i % n_colors]
            fig.add_trace(go.Scatter(
                x=t,
                y=v,
                mode='lines',
                name=str(neuron_id),
                line=dict(color=color, width=2)
            ))

        fig.update_layout(
            title='Network Activity',
            xaxis_title='Time (ms)',
            yaxis_title='Voltage (mV)',
            template='plotly_dark',
            hovermode='x unified',
            width=1000,
            height=600
        )
        for stim_time in stim_times:
            fig.add_vline(x=stim_time, line_dash='dash', line_color='white', opacity=0.5)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if save_path.endswith(".html"):
                fig.write_html(save_path)
            elif save_path.endswith(".png"):
                fig.write_image(save_path)
            else:
                raise ValueError("❌ Unknown format for interactive 2D plot — use .html or .png")
        else:
            fig.show()
        return

    # Matplotlib static 2D
    else:
        plt.figure(figsize=(12, 6))
        cmap = cm.get_cmap('viridis', len(voltages))

        for i, (neuron_id, v) in enumerate(voltages.items()):
            color = cmap(i)
            plt.plot(t, v, label=str(neuron_id), color=color, linewidth=2)

        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        plt.title('Network Activity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        for stim_time in stim_times:
            plt.axvline(x=stim_time, color='black', linestyle='--', alpha=0.5)
            
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
        return