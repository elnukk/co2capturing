# conda install basemap

%matplotlib inline
from enum import Enum
import imageio
from IPython.display import HTML
from itertools import chain
import math
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import os
import pandas as pd
from PIL import Image
from random import *
from sklearn.cluster import KMeans
from statistics import mean

def draw_map(m, scale=1.5):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    
    #latitudes and longitudes are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))
    
    #keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')


data=pd.read_csv('../input/glodapv2-ocean-data/GLODAPv2 Merged Master File.csv')
data


z_list=list(data["phtsinsitutp"])
zeta = list(data["tco2"])
x_lis=list(data["longitude"])
y_lis=list(data["latitude"])
input_data={'phtsinsituetp':z_list,'tco2':zeta,'longitude':x_lis,'latitude':y_lis}



data=''
for i in range(0,15259):
    data += str(str(input_data['longitude'][i])+" "+str(input_data['latitude'][i])+"; ")
unused=np.array(np.mat(data[:-2]))
tco2 = np.array(input_data['tco2'])



model = KMeans(n_clusters=381)
model.fit(unused)


model.cluster_centers_


label=np.array(model.labels_)


sse = []
for K in range(100, 1000, 100):
    model = KMeans(n_clusters=K)
    model.fit(unused)
    sse.append(model.inertia_)


  
plt.plot(range(100, 1000, 100), sse)


fig = plt.figure(figsize=(8, 6), edgecolor='w')
m = Basemap(projection='moll', lon_0=0, lat_0=0)
u_labels = np.unique(label)
m.drawmapboundary(fill_color='#76a6cc')
m.fillcontinents(color='white',lake_color='#76a6cc')
m.drawcoastlines()

#plotting the results:
for i in u_labels:
    x, y = m(unused[label == i , 0] , unused[label == i , 1])
    m.scatter(x, y , label = i)
plt.show()



set = randint(0,381)



fig = plt.figure(figsize=(8, 6), edgecolor='w')
m = Basemap(projection='moll', lon_0=0, lat_0=0)
m.drawmapboundary(fill_color='#76a6cc')
m.fillcontinents(color='white',lake_color='#76a6cc')
m.drawcoastlines()
x, y = m(unused[label==set, 0], unused[label==set, 1])
m.scatter(x, y, marker='o', color="r")


plt.savefig('plot.png')
plt.show()


show_animation = True


class RobotType(Enum):
    circle = 0
    rectangle = 1



class Config:


    def __init__(self):
        # robot parameter
        self.max_speed = 2.0 # [m/s]
        self.min_speed = -2.0  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.05  # constant to prevent the robot from getting stuck
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 10.0  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check
        # obstacles [x(m) y(m), ....]
        self.ob = np.column_stack((np.random.randint(-180,high=180, size=50), np.random.randint(-90,high=90, size=50)))


    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


config = Config()


def dwa_control(x, config, goal, ob):

    dw = calc_dynamic_window(x, config)

    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)

    return u, trajectory


def motion(x, u, dt):


    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x

def calc_dynamic_window(x, config):


    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw

def predict_trajectory(x_init, v, y, config):


    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory

def calc_control_and_trajectory(x, dw, config, goal, ob):
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            trajectory = predict_trajectory(x_init, v, y, config)
            # calc cost
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

            final_cost = to_goal_cost + speed_cost + ob_cost

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    # to ensure the robot do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_trajectory

def calc_obstacle_cost(trajectory, ob, config):

    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("Inf")

    min_r = np.min(r)
    return 1.0 / min_r  # OK

def calc_to_goal_cost(trajectory, goal):

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost


def plot_robot(x, y, yaw, config):  # pragma: no cover
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], marker="_",color="g")

def plot_arrow(x, y, yaw, length=5, width=1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)

def predict_PHB_yield(tco2):
    return (tco2*2.15225*10**11)/1461000

def main(initial, goal_x, goal_y, n_pts, robot_type=RobotType.circle):
    print(" start!!")
    PHB_yield=0
    num=0
    
    for i in range(n_pts):
        # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        x = np.array([initial[0], initial[1], math.pi / 12.0, 0.0, 0.0])
        # goal position [x(m), y(m)]
        goal = np.array([goal_x[i], goal_y[i]])

        # input [forward speed, yaw_rate]

        config.robot_type = robot_type
        trajectory = np.array(x)
        ob = config.ob
        while True:
            u, predicted_trajectory = dwa_control(x, config, goal, ob)
            x = motion(x, u, config.dt)  # simulate robot
            final_trajectory = np.vstack((trajectory, x))  # store state history
            #previous = np.vstack(initial)
           

            
            if show_animation:
                plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1],"-g")
                plt.plot(x[0], x[1],"xr")
                plt.plot(goal_x, goal_y, "xb")
                #plt.plot(previous[:, 0], previous[:, 1], "Dg")
                plt.plot(ob[:, 0], ob[:, 1], "ok")
                plt.plot(final_trajectory[:, 0], final_trajectory[:, 1],"-r")
                plot_robot(x[0], x[1], x[2], config)
                plot_arrow(x[0], x[1], x[2])
                plt.axis("equal")
                plt.grid(True)
                plt.xlabel('Degrees of Longitude')
                plt.ylabel('Degrees of Latitude')
                plt.savefig('Pictures/plot_{}'.format(num))
                
                plt.pause(1)
                
            # check reaching goal
            dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
            names.append(Image.open('/kaggle/working/Pictures/plot_{}.png'.format(num)))
            
            num+=1
            if dist_to_goal <= config.robot_radius:
                PHB_yield+=predict_PHB_yield(tco2[x_lis.index(goal[0])])
                print(f"Target Reached!!\n{PHB_yield} kg of PHB produced over the course of the trip")
                initial=goal
                break

    print("Done")
    
    plt.show()

if __name__ == '__main__':
    main([randint(
        math.floor(np.amin(unused[label==set], axis=0)[0]), 
        math.ceil(np.amax(unused[label==set], axis=0)[0])), randint(
        math.floor(np.amin(unused[label==set], axis=0)[1]),
        math.ceil(np.amax(unused[label==set], axis=0)[1]))],unused[label==0, 0],unused[label==0, 1],1, robot_type=RobotType.circle)
