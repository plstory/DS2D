import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format

RENAME_VARIABLES = {
        'total_area_self_consistency': 'Total Area (SC)',
        'polygon_area_self_consistency': 'P. Area (SC)',
        'room_count_self_consistency': 'R. Count (SC)',
        'room_id_self_consistency': 'R. ID (SC)',
        'polygon_overlap_count_self_consistency': 'P. Overlap (SC)',
        'room_height_self_consistency': 'R. Height (SC)',
        'room_width_self_consistency': 'R. Width (SC)',
        'num_room_prompt_consistency': 'Num. R. (PC)',
        'room_id_prompt_consistency': 'R. ID (PC)',
        'room_area_prompt_consistency': 'R. Area (PC)',
        'polygon_area_sum_vs_total_area_prompt_consistency': 'P. Area vs Total Area (PC)',
        'room_type_prompt_consistency': 'R. Type (PC)',
        'room_id_type_match_prompt_consistency': 'R. IDvsType (PC)',
        'room_height_prompt_consistency': 'R. Height (PC)',
        'room_width_prompt_consistency': 'R. Width (PC)',
        'json_file_consistency': 'JSON File (SC)',
    }
CATEGORIES = ['json_file_consistency', 'total_area_self_consistency', 'polygon_overlap_count_self_consistency', 
                      'room_count_self_consistency', 'room_width_self_consistency', 'room_height_self_consistency',
                      'polygon_area_self_consistency', 'polygon_area_sum_vs_total_area_prompt_consistency',
                      'room_area_prompt_consistency', 'room_id_type_match_prompt_consistency', 'num_room_prompt_consistency', 
                      'room_id_prompt_consistency', 'room_type_prompt_consistency']

def plot_categories_sanity_check(strat_lookup, categories):

    for strat, summary in strat_lookup.items():
        for category in categories:
            if isinstance(summary, dict):
                metric_result = summary[category]
            else:
                metric_result = getattr(summary, category)
            if metric_result is None or category == 'json_strict_file_consistency':
                categories.remove(category)
    return categories

def get_df_from_summary(summary, categories=None, strat_name='Dataset'):
    
    if categories is None:
        categories = CATEGORIES
    
    r = []
    theta = []
    for category in categories:
        if isinstance(summary, dict):
            metric_result = summary[category]
        else:
            metric_result = getattr(summary, category)
        if metric_result is None or category == 'json_strict_file_consistency':
            continue
        elif category in ['polygon_area_self_consistency']:
            r.append(1-metric_result[0])
        elif category in ['room_id_prompt_consistency', 'room_type_prompt_consistency']:
            r.append(metric_result[1])
        elif category in ['room_width_self_consistency', 'room_height_self_consistency', 'polygon_area_sum_vs_total_area_prompt_consistency',
                        'room_area_prompt_consistency', 'num_room_prompt_consistency', 'total_area_self_consistency',
                        'room_height_prompt_consistency', 'room_width_prompt_consistency']:
            r.append(1-metric_result)
        else:
            r.append(metric_result)
        theta.append(RENAME_VARIABLES[category])
    
    if not r:
        return None
    ret = pd.DataFrame(dict(r=r, theta=theta))
    ret['Model'] = strat_name
    return ret

def get_df_from_summary_std(summary, categories=None, strat_name='Dataset'):
    
    if categories is None:
        categories = CATEGORIES
    
    r = []
    theta = []
    for category in categories:
        if isinstance(summary, dict):
            metric_result = summary[category]
        else:
            metric_result = getattr(summary, category)
        if metric_result is None or category == 'json_strict_file_consistency':
            continue
        elif category in ['polygon_area_self_consistency']:
            r.append(metric_result[0])
        elif category in ['room_id_prompt_consistency', 'room_type_prompt_consistency']:
            r.append(metric_result[1])
        else:
            r.append(metric_result)
        theta.append(RENAME_VARIABLES[category])
    
    if not r:
        return None
    ret = pd.DataFrame(dict(r=r, theta=theta))
    ret['Model'] = strat_name
    return ret

def get_df_from_summary_v2(summary, categories=None, strat_name='Dataset', if_std=False):
    df = get_df_from_summary(summary, categories, strat_name=strat_name) if not if_std \
        else get_df_from_summary_std(summary, categories, strat_name=strat_name)
    df = df.rename(index={i:metric for i, metric in enumerate(df['theta'])})
    df = df.drop(columns=['Model','theta'])
    df = df.rename(columns={'r':strat_name}).T
    return df

def get_df_from_summary_separated_by_num_rooms(summaries_by_room, categories=None, strat_name='Dataset'):
    if categories is None:
        categories = CATEGORIES
    
    ret = None
    for num_rooms, summary in summaries_by_room.items():
        df_by_num_room = get_df_from_summary(summary, categories, strat_name)
        df_by_num_room = df_by_num_room.rename(index={i:metric for i, metric in enumerate(df_by_num_room['theta'])})
        df_by_num_room = df_by_num_room.drop(columns=['Model','theta'])
        df_by_num_room = df_by_num_room.rename(columns={'r':num_rooms})
        ret = df_by_num_room if ret is None else pd.concat([ret, df_by_num_room], axis=1)
    return ret


def get_df_from_summary_separated_by_num_rooms_v2(summaries_by_room, strat_name='Dataset'):

    ret = None
    for num_rooms, (summary, categories) in summaries_by_room.items():
        df_by_num_room = get_df_from_summary(summary, categories, strat_name)
        df_by_num_room = df_by_num_room.rename(index={i:metric for i, metric in enumerate(df_by_num_room['theta'])})
        df_by_num_room = df_by_num_room.drop(columns=['Model','theta'])
        df_by_num_room = df_by_num_room.rename(columns={'r':num_rooms})
        ret = df_by_num_room if ret is None else pd.concat([ret, df_by_num_room], axis=1)
    return ret.sort_index(axis=1)

def get_latex_from_df(df):
    df.sort_index(axis=1, inplace=True)
    df = df.style.format(precision=3)
    return df.to_latex()

def plot_3d_from_df(df, filename, title=None):

    df.sort_index(axis=1, inplace=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    
    # Thickness of the bars
    dx, dy = 0.5, 0.5

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set up positions for the bars
    xpos = np.arange(df.shape[0])
    ypos = np.arange(df.shape[1])

    # Create meshgrid
    xpos, ypos = np.meshgrid(xpos, ypos)
    xpos = xpos.flatten()
    ypos = ypos.flatten()

    # The bars start from 0 altitude
    zpos = np.zeros(df.shape).flatten()

    # The bars' heights
    dz = df.values.ravel()

    # Plot
    values = np.linspace(0.2, 1., xpos.shape[0])
    colors = cm.rainbow(values)
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, alpha=0.8, color=colors)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Number of rooms')
    ax.set_zlabel('Accuracies')

    # Set the ticks in the middle of the bars
    ax.set_xticks(np.arange(len(df.index)) + dx / 2)
    ax.set_yticks(np.arange(len(df.columns)) + dy / 2)

    # Set tick labels
    ax.set_xticklabels(df.index, ha='left')
    ax.set_yticklabels(df.columns[::-1], rotation=45, ha='right', rotation_mode='anchor')
    if title:
        ax.set_title(title)

    # # Ensure tight layout
    # plt.tight_layout()

    # Save the figure
    plt.savefig(filename)

    # Close the figure to free memory
    plt.close(fig)


def plot_radar_from_df(df, filename, title=None):
    import plotly.express as px
    from pathlib import Path

    fig = px.line_polar(df, r='r',
                            theta='theta',
                            color='Model',
                            line_close=True,
                            title=title)
    fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1.0]
                    )),
                showlegend=True,
                template='plotly_white'
            )
    fig.write_image(Path(filename))
    