from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from typing import Tuple, Union, List
import re

import numpy as np
import pandas as pd
import torch
from einops import repeat

from tensordict.tensordict import TensorDict
from rl4co.models import MatNetPolicy, MatNet, PPO, AttentionModel, POMO, AttentionModelPolicy
from rl4co.utils import RL4COTrainer

from rl4co.envs.routing.evrptw.generator import EVRPGenerator
from rl4co.envs.routing.evrptw.env import EVRPEnv
from rl4co.utils.ops import get_distance_matrix
from rl4co.utils.pylogger import get_pylogger
from rl4co.models.zoo import DeepACO


def read_instance(file_path):
    instance = {
        'nodes': [],
        'vehicletypes': [],
        'arcs': [],
        'num_nodes': 0  # We'll determine this based on the NODES section
    }

    with open(file_path, 'r') as f:
        section = None

        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Identify the sections
            if line.startswith('NODES'):
                instance['num_nodes'] = int(line.split()[1])  # Extract number of nodes
                section = 'nodes'
                continue
            elif line.startswith('VEHICLETYPES'):
                section = 'vehicletypes'
                continue
            elif line.startswith('ARCS'):
                section = 'arcs'
                continue

            # Parse nodes section
            if section == 'nodes':
                parts = line.split()
                if len(parts) == 5:
                    node = {
                        'type': parts[0],  # Node type (d, f, c)
                        'id': int(parts[1]),  # Node id
                        'demand': int(parts[2]),  # Demand
                        'earliest': int(parts[3]),  # Earliest time
                        'latest': int(parts[4])  # Latest time
                    }
                    instance['nodes'].append(node)

            # Parse vehicle types section
            elif section == 'vehicletypes':
                parts = line.split()
                if len(parts) == 7:
                    vehicle = {
                        'start_node': int(parts[0]),  # Start node id
                        'end_node': int(parts[1]),  # End node id
                        'Hmax': int(parts[2]),  # Maximum capacity
                        'Hinit': int(parts[3]),  # Initial capacity
                        'Hfinal': int(parts[4]),  # Final capacity
                        'capacity': int(parts[5]),  # Vehicle capacity
                        'num_vehicles': int(parts[6])  # Number of vehicles
                    }
                    instance['vehicletypes'].append(vehicle)

            # Parse arcs section
            elif section == 'arcs':
                parts = line.split()
                if len(parts) == 5:
                    arc = {
                        'tail': int(parts[0]),  # Tail node id
                        'head': int(parts[1]),  # Head node id
                        'cost': int(parts[2]),  # Cost
                        'time': int(parts[3]),  # Time
                        'energy': int(parts[4])  # Energy consumption
                    }
                    instance['arcs'].append(arc)

    # Create the cost, time, and energy matrices
    num_nodes = instance['num_nodes']
    cost_matrix = np.full((num_nodes, num_nodes), 0.0)  # Initialize with infinity (no direct path)
    time_matrix = np.full((num_nodes, num_nodes), 0.0)  # Same for time
    energy_matrix = np.full((num_nodes, num_nodes), 0.0)  # Same for energy

    factor = 100
    # Fill the matrices with values from the arcs
    for arc in instance['arcs']:
        tail = arc['tail']
        head = arc['head']
        cost_matrix[tail][head] = arc['cost']  # / 100
        time_matrix[tail][head] = arc['time']  # / 100
        energy_matrix[tail][head] = arc['energy']  # / 100

    return instance, cost_matrix, time_matrix, energy_matrix


@dataclass
class Stop:
    id: int
    type: str  # 'D' for depot, 'C' for customer, 'R' for recharging
    service_time: int


@dataclass
class Route:
    route_id: int
    vehicle_id: int  # Extracted from <X>
    stops: List[Stop]
    cost: float


@dataclass
class Solution:
    instance: str
    available_vehicles: float
    num_customers: float
    recharging_strategy: str  # BS, FR, or PR
    objective: str  # DI or CT
    is_optimal: bool
    run_time: float
    run_time_root: float
    gap_percent: float  # Gap in percent
    total_distance: float
    completion_time: float
    routes: List[Route]


def parse_route(route_str: str) -> Route:
    # Extract route ID and vehicle ID
    route_match = re.match(r'(\d+):\s*<(\d+)>\s*\((.*)\)\s*cost\s*=\s*(\d+)', route_str)
    if not route_match:
        raise ValueError(f"Invalid route format: {route_str}")

    route_id, vehicle_id, stops_str, cost = route_match.groups()

    # Parse stops
    stops = []
    stop_pattern = re.compile(r'(\d+)\(([CDR]),(\d+)\)')
    for stop_match in stop_pattern.finditer(stops_str):
        stop_id, stop_type, service_time = stop_match.groups()
        stops.append(Stop(
            id=int(stop_id),
            type=stop_type,
            service_time=int(service_time)
        ))

    return Route(
        route_id=int(route_id),
        vehicle_id=int(vehicle_id),
        stops=stops,
        cost=float(cost)
    )


def parse_csv_line(line: str) -> Solution:
    fields = line.strip().split(';')

    # Parse routes
    routes_str = fields[-1]
    routes = []
    for route in routes_str.split(' , '):
        if route.strip():
            routes.append(parse_route(route.strip()))

    return Solution(
        instance=fields[0],
        available_vehicles=float(fields[1]),
        num_customers=float(fields[2]),
        recharging_strategy=fields[3],
        objective=fields[4],
        is_optimal=fields[5].lower() == 'true',
        run_time=float(fields[6]),
        run_time_root=float(fields[7]),
        gap_percent=float(fields[8]),  # Already in percent according to readme
        total_distance=float(fields[9]) if fields[9] else '',
        completion_time=float(fields[10]) if fields[10] else '',
        routes=routes
    )


def read_solution(file_path: str) -> List[Solution]:
    solutions = []
    with open(file_path, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            if line.strip():
                solutions.append(parse_csv_line(line))
    return solutions

# Example usage
instance, cost_matrix, time_matrix, energy_matrix = read_instance(
    './ejor/ejor/instances/karis/karis_S_empty_fast_1_100.inst')

solutions = read_solution('./ejor/ejor/solutions/karis/KarisSolutions.csv')
bs_solutions = []
for solution in solutions:
    if solution.recharging_strategy == "BS" and solution.total_distance:
        bs_solutions.append(solution)
n_instances = len(bs_solutions)
n = 1
bs_solutions = [bs_solutions[0]]
for bs_solution in bs_solutions:
    print(f"checking instance {n} / {n_instances}")
    instance, cost_matrix, time_matrix, energy_matrix = read_instance(
        f'./ejor/ejor/instances/karis/{bs_solution.instance}')
    # Access matrices
    # print("Cost Matrix:\n", cost_matrix)
    # print("Time Matrix:\n", time_matrix)
    # print("Energy Matrix:\n", energy_matrix)
    start_nodes = []
    end_nodes = []
    for v in instance["vehicletypes"]:
       start_nodes.append(v["start_node"])
       end_nodes.append(v["end_node"])

    n_customers = 0
    for node in instance["nodes"]:
        if node["type"] == "c":
            n_customers += 1
    print(n_customers)
    n_vehicles = len(instance["vehicletypes"])

    h_init = instance["vehicletypes"][0]["Hinit"] / 1000
    h_max = instance["vehicletypes"][0]["Hmax"] / 1000
    h_final = instance["vehicletypes"][0]["Hfinal"] / 1000

    if torch.cuda.is_available():
        accelerator = "gpu"
        batch_size = 256
        train_data_size = 1_000
        embed_dim = 128
        num_encoder_layers = 4
    else:
        accelerator = "cpu"
        batch_size = 16
        train_data_size = 10_000
        embed_dim = 128
        num_encoder_layers = 2
    # cm = torch.tensor(cost_matrix).float() / 1000
    # cm = repeat(cm, 'n d -> b n d', b=batch_size, d=cm.shape[0])
    env = EVRPEnv(generator_params={"instance_dm": cost_matrix,
                                    "n_customers": n_customers,
                                    "min_num_agents": n_vehicles,
                                    "max_num_agents": n_vehicles,
                                    "h_init": h_init,
                                    "h_max": h_max,
                                    "h_final": h_final,
                                    "dmat_only": True})

    policy = MatNetPolicy()
    model = MatNet(env=env,
                   policy=policy,
                   baseline="shared",
                   batch_size=batch_size,
                   train_data_size=train_data_size,
                   val_data_size=2_000)
    td_init = env.reset(batch_size=[batch_size])

    trainer = RL4COTrainer(max_epochs=1, devices="auto")
    trainer.fit(model)

    out = model.policy(td_init.clone(), env, phase="test", decode_type="beam_search", return_actions=True)
    print(out["reward"])
    for td, actions in zip(td_init, out['actions'].cpu()):
        fig = env.render(td, actions)
        fig.show()

    actions = []
    for route in bs_solution.routes:
        for stop in route.stops:
            # if stop.id not in start_nodes:
            actions.append(stop.id)
    # for end_node in end_nodes:
    #     if end_node not in actions:
    #         actions.append(end_node)

    batch_size = 2
    td_eval = env.reset(batch_size=[batch_size]).to("cpu")
    cm = torch.tensor(cost_matrix).float() #/ 1000
    em = torch.tensor(energy_matrix).float() #/ 1000
    td_eval["cost_matrix"] = repeat(cm, 'n d -> b n d', b=batch_size, d=td_eval["cost_matrix"].shape[1])
    td_eval["energy_consumption"] = repeat(em, 'n d -> b n d', b=batch_size, d=td_eval["cost_matrix"].shape[1])
    actions = torch.tensor(actions)
    fig, result = env.render(td_eval[0], actions)
    try:
        assert result == bs_solution.total_distance
        print(f"instance {bs_solution.instance} OK, {result} == {bs_solution.total_distance}")
    except:
        print(f"instance {bs_solution.instance} Failed")
        print(result, bs_solution.total_distance)
    n += 1
    # fig.show()
