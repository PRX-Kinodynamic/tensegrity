import heapq
import numpy as np
import random
from close_node import is_point_within_distance
from util_heuristic import dist_heuristic, l2_dist, coll_det, fill_grid, snap_to_grid


def rel_mov(x, y, theta):
    # Calculate the change in x and y using trigonometry
    dx = x * np.cos(theta) - y * np.sin(theta)
    dy = x * np.sin(theta) + y * np.cos(theta)
    return dx, dy

def angle_norm(x):
    return x % (2*np.pi)

def heuristic(a, b, obstacles, heur_type, grid_step, k=0,grid = []):
    if heur_type == "wave":
        approx = snap_to_grid(a[:2],grid_step)
        if approx in grid:
            return grid[approx]
        else:
            return np.inf
    else:
        return dist_heuristic(a[:2], b[:2], obstacles, k)





# A* algorithm implementation
def astar(start, goal, gaits, obstacles=(),tolerance = 0.1, rot_tol = np.pi/4, repeat_tol = 0.07, single_push = False, stochastic = True,\
          heur_type = "dist", boundary = (-1,1,-1,1), obstacle_dims=(0.4,0.27), grid_step = 0.01):
    if heur_type == "wave":
        h = fill_grid(goal[:2], boundary, grid_step, obstacles=obstacles, obstacle_size=obstacle_dims)
    else:
        h = {}
    # Initialize open and closed lists
    open_list = []
    heapq.heappush(open_list, (0, start, -1))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal, obstacles, heur_type, grid_step, grid = h)}
    closed_list = [] #only used to check if already visited using kd tree

    while open_list:
        # Get the node with the lowest f_score value
        node = heapq.heappop(open_list)
        current = node[1]

        if coll_det(current, obstacles, boundary=boundary, obstacle_dims=obstacle_dims) or (len(closed_list)>0 and is_point_within_distance(current[:2], closed_list, repeat_tol)):
                closed_list.append(current[:2])
                continue

        if not single_push:
            closed_list.append(current[:2])

        # If the goal is reached, reconstruct and return the path
        if l2_dist(current[:2], goal[:2])<= tolerance and min(abs(angle_norm(current[2]) - goal[2]), np.pi - abs(angle_norm(current[2]) - goal[2])) <= rot_tol :
            path = []
            movements = []
            while current in came_from:
                prev = came_from[current][0]
                move = came_from[current][1]
                path.append(current)
                movements.append(move)
                current = prev
            path.append(start)
            return path[::-1], movements[::-1]


        if single_push:
            if stochastic:
                each_neighbor = []
            else:
                best_neighbor = None
        # best_f = 0
        for k in range(len(gaits)):
            gait_num = gaits[k]
            dx , dy = rel_mov(gait_num[0], gait_num[1], current[2])
            neighbor = (current[0]+dx, current[1]+dy, angle_norm(current[2]+gait_num[2]))
            
            tentative_g_score = g_score[current] + np.sqrt(dx**2 + dy**2)
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = (current, k)
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor[:2], goal, obstacles, heur_type, grid_step, grid = h)
                if single_push:
                    if stochastic:
                        each_neighbor.append((f_score[neighbor], neighbor, k))
                    else:
                        if best_neighbor == None or f_score[neighbor] < best_neighbor[0]:
                            best_neighbor = (f_score[neighbor], neighbor, k)
                else:
                    heapq.heappush(open_list, (f_score[neighbor], neighbor, k))

        if single_push:
            if stochastic:
                if len(each_neighbor)==0:
                    closed_list.append(current[:2])
                    continue
                elif len(each_neighbor)==1:
                    heapq.heappush(open_list, each_neighbor[0])
                beta = 1.0
                weights = [np.exp(-beta * node[0]) for node in each_neighbor]
                total_weight = sum(weights)
                probabilities = [weight / total_weight for weight in weights]
                
                # Select a node based on the probabilities
                best_neighbor = random.choices(each_neighbor, weights=probabilities, k=1)[0]
                heapq.heappush(open_list, best_neighbor)
                heapq.heappush(open_list, node)

            else:
                if best_neighbor != None:
                    heapq.heappush(open_list, best_neighbor)
                    heapq.heappush(open_list, node)
                else:
                    closed_list.append(current[:2])
        
              


    # Return empty path if no path found
    print("Can't Find Path")
    return [], []
