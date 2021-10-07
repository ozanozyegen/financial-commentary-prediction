import numpy as np
import plotly.graph_objects as go

def xai_plot(results, config, title, 
    SAVE_DIR=None, plot_name='cam', color_scale='blues',):
    if config['DATASET'] in ['variance_data', 'rule_based']:
        y_cols_prefix = 'M'
    else:
        y_cols_prefix = ''
    zmin = zmax = zmid = None
    if color_scale == 'blues':
        zmin = 0
        zmax = np.max(results)
        zmid=None
    elif color_scale == 'RdBu':
        zmid=0

    data_cols = [f'{y_cols_prefix}{i}' for i in range(-config['HISTORY_SIZE'],0)]
    programmers = config['LABELS']
    z = np.round(results, 3)

    fig = go.Figure(data=go.Heatmap(
            z=z,
            x=programmers,
            y=data_cols,
            zmin=zmin,
            zmid=zmid,
            zmax=zmax,
            colorscale=color_scale))
    fig.update_layout(width=600, height=600, xaxis_showgrid=False,
        yaxis_showgrid=False, template='none')
    
    fig.update_layout(
        title=title,
        yaxis=dict(tickmode='linear',),
    )
    
    fig.show()
    if SAVE_DIR:
        fig.update_layout(title='')
        fig.write_image(f"{SAVE_DIR}/{config['DATASET']}_{plot_name}.pdf")