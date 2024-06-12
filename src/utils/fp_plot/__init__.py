import matplotlib.pyplot as plt
from . import procthorpy
# import rplanpy

# def plot_rplan(file: str, out_file: str = 'output_graph.png', plot_graph: bool = False) -> None:
#     data = rplanpy.data.RplanData(file)
#     ncols = 2 if plot_graph else 1
#     _fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols*5, 5))
    
#     if plot_graph:
#         rplanpy.plot.plot_floorplan(data, ax=ax[0], title="Rooms")
#         rplanpy.plot.plot_floorplan_graph(
#             data=data, with_colors=True, edge_label='door', ax=ax[1],
#             title="Bubble graph"
#         )
#     else:
#         rplanpy.plot.plot_floorplan(data, ax=ax, title="Rooms")

#     plt.tight_layout()
#     plt.savefig(out_file)
#     plt.show()

def plot_procthor(data, out_file: str = 'output_procthor.png') -> None:
    data = data["rooms"]
    ncols = 1
    _fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols*5, 5))
    
    procthorpy.plot.plot_floorplan(data, ax=ax, title=None, label_rooms=False)

    plt.tight_layout()
    plt.savefig(out_file,bbox_inches='tight', transparent=True)
    plt.clf()
    plt.close()
    # plt.show()