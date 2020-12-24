"""
Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)
"""

# importing libraries 
import math
import random
import matplotlib.pyplot as plt
import numpy as np

show_animation = True # variable for showing animation

# RRT class
class RRT:
    """
    Class for RRT planning
    """

    # new class used for describing a 'Node' of RRT
    class Node:
        """
        RRT Node
        """

        # initializing varibles for node
        def __init__(self, x, y):
            self.x = x 
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None # used for parent node

    # initializing variables for RRT class
    # start:Start Position [x,y]
    # goal:Goal Position [x,y]
    # obstacleList:obstacle Positions [[x,y,size],...]
    # randArea:Random Sampling Area [min,max]  
    def __init__(self, start, goal, obstacle_list, rand_area, max_expand_distance=3.0, path_resolution=0.5, min_goal_distance=5, max_iter=500):
        self.start = self.Node(start[0], start[1]) # start coordinates
        self.end = self.Node(goal[0], goal[1]) # goal coordinates 
        self.min_rand = rand_area[0] # min range for generating random number
        self.max_rand = rand_area[1] # max range for generating random number
        self.max_expand_distance = max_expand_distance # max distance for connecting a new node 
        self.path_resolution = path_resolution # resolution for drawing a path 
        self.min_goal_distance = min_goal_distance # min distance for sampling goal node
        self.max_iter = max_iter # total number of iterations
        self.obstacle_list = obstacle_list # coordinate of obstacles and their radius
        self.node_list = []

    # function for path planning 
    # animation: flag for animation on or off
    def path_planning(self, animation=True):

        self.node_list = [self.start]

        for i in range(self.max_iter):
            rnd_node = self.get_random_node() # generating random node
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node) # finding the index of nearest node to new random node
            nearest_node = self.node_list[nearest_ind] # geting the value of nearest node

            new_node = self.move_to_node(nearest_node, rnd_node, self.max_expand_distance) # moving to the new node

            if self.check_collision(new_node, self.obstacle_list): # checking for collision
                self.node_list.append(new_node) # if no obstacles then add new node

            # ploting on graph
            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            # if the distance between latest node and goal is less than max expand distance then connect with goal node
            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.max_expand_distance:
                final_node = self.move_to_node(self.node_list[-1], self.end, self.max_expand_distance)
                
                if self.check_collision(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5:
                self.draw_graph(rnd_node)

        return None  # cannot find path

    # movingto node function
    def move_to_node(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        # choose whichever value is less between d, extend_length
        if extend_length > d:
            extend_length = d

        # divide distance by the path resolution to go in steps 
        # this is usefull for reverse track
        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        
        # if close to the new random node then append
        if d <= self.path_resolution:
            new_node.x = to_node.x
            new_node.y = to_node.y
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)

        new_node.parent = from_node # node from which it is connected

        return new_node

    # for connecting start with goal node
    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]] 
        node = self.node_list[goal_ind]

        # # starting from goal till the start node append in a list 
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy) # will return hypotenous distance between to coordinates

    # for generating random nodes
    def get_random_node(self):
        if random.randint(0, 100) > self.min_goal_distance: # if not in the vicinity of goal coordinate
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    # for plotting the graph
    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.05)

    # using static method for the obstacles
    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    # to get nearest node
    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    # function to check for obstacles in path
    @staticmethod
    def check_collision(node, obstacleList):

        if node is None:
            return False

        # will check distance between each coordinate of obstacle and each coordinate in the path
        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            # if the min distance between obstacle coordinat and coordinate in the path
            # is less than the radius of the obstacle the return false i.e. collision is detected
            if min(d_list) <= size**2:
                return False  # collision

        return True  # if not then it is safe

    # to calculate distance and angle between 2 nodes
    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

# main program
def main(gx=14.0, gy=14.0):
    print("start " + __file__)

    # ====Search Path with RRT====
    # obstacle list in format: [x, y, radius]
    obstacleList = [(10,14,2), (5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2), (9, 5, 2), (8, 10, 1)]
    
    # defining instance of RRT class
    rrt = RRT(
        start=[0, 0],
        goal=[gx, gy],
        rand_area=[-2, 15],
        obstacle_list=obstacleList)

    path = rrt.path_planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()


if __name__ == '__main__':
    main()