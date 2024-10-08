import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from matplotlib import colormaps

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td, actions=None, ax=None):
    def discrete_cmap(num, base_cmap="nipy_spectral"):
        """Create an N-bin discrete colormap from the specified input map"""
        base = colormaps[base_cmap]
        color_list = base(np.linspace(0, 1, num))
        cmap_name = base.name + str(num)
        return base.from_list(cmap_name, color_list, num)

    if actions is None:
        actions = td.get("action", None)
    # if batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        if actions.dim() > 1:
            actions = actions[0]
    num_agents = td["num_agents"]
    num_points = td["customer_nodes"].shape[-1]
    # locs = td["locs"]
    cmap = discrete_cmap(num_agents, "rainbow")

    fig, ax = plt.subplots()
    cols = int(np.ceil(np.sqrt(num_points)))
    rows = int(np.ceil(num_points / cols))

    # Generate the rectangular spread coordinates
    rectangular_spread = [(x, y) for x in np.linspace(
        -1.5, 1.5, cols) for y in np.linspace(-1.5, 1.5, rows)]
    rectangular_spread = rectangular_spread[:num_points]
    max_coord = max(rectangular_spread)
    y_cs = max_coord[0] + 1.5
    # Limit to the number of points needed
    top_point = [(0, y_cs)]

    y_values = np.linspace(0.5, -1, num_agents)
    left_points = [(-3, y) for y in y_values]  # Points going down on the left
    right_points = [(3, y) for y in y_values]
    # Updating all points
    all_points = rectangular_spread + top_point + left_points + right_points

    # Plotting the updated points
    #plt.figure(figsize=(8, 8))
    x, y = zip(*all_points)
    plt.scatter(x, y, color='blue')

    # Annotating the points with their indices
    for i, (x_coord, y_coord) in enumerate(all_points):
        plt.text(x_coord, y_coord, str(i), fontsize=12, ha='right', color='red')

    # Setting axis limits and labels for clarity
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.title("21 Points with Rectangular Spread in the Middle")
    plt.grid(True)

    # Add depot action = 0 to before first action and after last action
    # actions = torch.cat(
    #     [
    #         torch.zeros(1, dtype=torch.int64),
    #         actions,
    #         torch.zeros(1, dtype=torch.int64),
    #     ]
    # )

    start_depots = td["start_depots"]
    end_depots = td["end_depots"]
    charging_node = td["charging_node"]

    # Make list of colors from matplotlib
    for i, loc in enumerate(all_points):
        if i in td["end_depots"].numpy():
            # depot
            marker = "s"
            color = "g"
            label = "End Depot"
            markersize = 10
        elif i in td["start_depots"].numpy():
            # depot
            marker = "s"
            color = "g"
            label = "Start Depot"
            markersize = 10
        elif i == charging_node.item():
            # cs
            marker = "s"
            color = "r"
            label = "CS"
            markersize = 10
        else:
            # normal location
            marker = "o"
            color = "tab:blue"
            label = "Customers"
            markersize = 8
        if i > 1:
            label = ""

        ax.plot(
            loc[0],
            loc[1],
            color=color,
            marker=marker,
            markersize=markersize,
            label=label,
        )
    actions_result = []

    actions_result.append(start_depots[0])
    start_depots = start_depots[1:]
    for action in actions:
        actions_result.append(action)
        if action in end_depots and start_depots.size()[0] > 0:
            actions_result.append(start_depots[0])
            start_depots = start_depots[1:]
    if start_depots.size()[0] == 0 and end_depots[-1] not in actions_result:
        actions_result.append(end_depots[-1])

    # Plot the actions in order
    agent_idx = 1
    agent_travel_time = 0
    max_travel = 0
    n_charges = {i+1: 0 for i in range(num_agents)}
    battery_levels = {i+1: [100] for i in range(num_agents)}

    for i in range(len(actions_result)):
        if actions_result[i] in end_depots:
            agent_idx += 1
            idx = list(end_depots.numpy()).index(actions_result[i].item())
            ax.annotate(str(agent_travel_time), right_points[idx])
            if agent_travel_time > max_travel:
                max_travel = agent_travel_time
            agent_travel_time = 0
            continue

        color = cmap(num_agents - agent_idx)

        from_node = actions_result[i]
        to_node = actions_result[i + 1]
        # to_node = (
        #     actions_result[i + 1] if i < len(actions_result) - 1 else actions_result[0]
        # )  # last goes back to depot
        from_loc = all_points[from_node]
        to_loc = all_points[to_node]
        travel_time = td["cost_matrix"][from_node, to_node]
        if agent_idx <= num_agents:
            battery_levels[agent_idx].append(battery_levels[agent_idx][-1] - 2 * travel_time)
        if to_node == charging_node.item():
            travel_time += 5
            n_charges[agent_idx] += 1
            battery_levels[agent_idx].append(torch.tensor([100]))
        cur_battery = str(round(battery_levels[agent_idx][-1].item()))
        agent_travel_time += travel_time
        ax.plot([from_loc[0], to_loc[0]], [from_loc[1], to_loc[1]], color=color)
        ax.annotate(
            "",
            xy=(to_loc[0], to_loc[1]),
            xytext=(from_loc[0], from_loc[1]),
            arrowprops=dict(arrowstyle="->", color=color),
            annotation_clip=False,
        )
    print(max_travel)
    print(n_charges)
    print(battery_levels)
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_title("mTSP")
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("y-coordinate")
    return fig