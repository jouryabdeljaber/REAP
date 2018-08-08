
import matplotlib.pyplot as plt
import numpy as np
import os
from pydrake.all import (BasicVector, DiagramBuilder, FloatingBaseType,
                         RigidBodyPlant, RigidBodyTree, Simulator, VectorSystem, SignalLogger)
from underactuated import (FindResource, PlanarRigidBodyVisualizer)

# Load the double pendulum from Universal Robot Description Format
#tree = RigidBodyTree(FindResource(os.path.dirname(os.path.realpath(__file__)) + "/block_pusher2.urdf"),
#                     FloatingBaseType.kFixed)





def IVK(q, v_man):
    l = 2
    return np.linalg.lstsq(np.array([[np.cos(q[0]) + np.cos(q[0] + q[1]), np.cos(q[0] + q[1])],[np.sin(q[0]) + np.sin(q[0] + q[1]), np.sin(q[0] + q[1])]]), v_man/l)[0]

def sigm(x):
    return 1./(1. + np.exp(-10000*x))



def R(theta):
    return np.array([[np.cos(theta),np.sin(theta)*-1], [np.sin(theta),np.cos(theta)]])

def findForce(v_W, q_theta, v_theta, G):
    
    V_O = R(-1*q_theta).dot(v_W)
    V_vector = np.array([V_O[0], V_O[1], v_theta])
    F_vector = G(V_vector)
    f_theta = F_vector[2]
    f_O = np.array([F_vector[0], F_vector[1]])
    f_W = R(q_theta).dot(f_O)
    return (f_W, f_theta) 

def G(V):
    A = -1 * V
    B = (1/np.linalg.norm(A))*A
    F = 0.1 * B
    return F

def H(F):
    number = (1/100)*np.linalg.norm(F)**2
    return number

class DController(VectorSystem):
    """ Defines a feedback controller for the double pendulum.

    The controller applies torques at the joints in order to
    1) cancel out the dynamics of the double pendulum,
    2) make the first joint swing with the dynamics of a single pendulum, and
    3) drive the second joint towards zero.

    The magnitude of gravity for the imposed single pendulum dynamics is taken
    as a constructor argument.  So you can do fun things like pretending that
    gravity is zero, or even inverting gravity!

    """

    def __init__(self, rigid_body_tree, B, v_des):
        # 4 inputs (double pend state), 2 torque outputs.
        self.tree = rigid_body_tree

        VectorSystem.__init__(self, self.tree.get_num_positions() + self.tree.get_num_velocities(), self.tree.get_num_actuators())
        
        self.B = B
        self.v_des = v_des
        self.num_arms = (self.tree.get_num_velocities() - 3) // 2

    def _DoCalcVectorOutput(self, context, double_pend_state, unused, torque):
        q = double_pend_state[:self.tree.get_num_positions()]
        v = double_pend_state[-self.tree.get_num_velocities():]
        V = v[-3:]
        torque[:] = 0.0*v
        f_W = np.zeros(2)
        f_theta = 0
        if (np.linalg.norm(V) > 0):
            (f_W, f_theta) = findForce(V[:2],q[-1],V[2],G)
        scale = 2*sigm(np.linalg.norm(V)) - 1
        f_W = f_W*scale
        f_theta = f_theta*scale
        torque[-3:-1] = f_W
        torque[-1] = f_theta
        v_des = self.v_des(context.get_time())
        u = 0.0*v_des
        for i in range(0,self.num_arms):
            inds = [2*i, 2*i + 1]
            u[inds] = IVK(q[inds], v_des[inds])
        torque[:(-3)] = self.B.dot(u - v[:(-3)])

class Player(VectorSystem):
    def __init__(self, data, times):
        VectorSystem.__init__(self, 0, data.shape[0])
        self.data = data
        self.times = times
        self.nt = times.size

    def _DoCalcVectorOutput(self, context, unused1, unused2, out):
        t = context.get_time()
        ind = 0
        if t <= self.times[0]:
            ind = 0
        elif t >= self.times[-1]:
            ind = self.nt - 1
        else:
            ind = np.argmax(self.times >= t)
        out[:] = self.data[:,ind]
        #import pdb; pdb.set_trace()







def simulateRobot(time, B, v_command):
    tree = RigidBodyTree(FindResource(os.path.dirname(os.path.realpath(__file__)) + "/block_pusher2.urdf"),
                    FloatingBaseType.kFixed)

    # Set up a block diagram with the robot (dynamics), the controller, and a
    # visualization block.
    builder = DiagramBuilder()
    robot = builder.AddSystem(RigidBodyPlant(tree))


    controller = builder.AddSystem(DController(tree, B, v_command))
    builder.Connect(robot.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), robot.get_input_port(0))

    Tview = np.array([[1., 0., 0., 0.],
                          [0., 1., 0., 0.],
                          [0., 0., 0., 1.]],
                         dtype=np.float64)
    visualizer = builder.AddSystem(PlanarRigidBodyVisualizer(tree,Tview,
                                                             xlim=[-2.8, 4.8],
                                                             ylim=[-2.8, 10]))
    builder.Connect(robot.get_output_port(0), visualizer.get_input_port(0))


    logger = builder.AddSystem(SignalLogger(tree.get_num_positions() + tree.get_num_velocities()))
    builder.Connect(robot.get_output_port(0), logger.get_input_port(0))

    diagram = builder.Build()

    # Set up a simulator to run this diagram
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(True)



    # Set the initial conditions
    context = simulator.get_mutable_context()
    state = context.get_mutable_continuous_state_vector()
    start1 = 3*np.pi/16
    start2 = 15*np.pi/16
    #np.pi/6 - eps, 2*np.pi/3 + eps, -np.pi/6 + eps, -2*np.pi/3 - eps,    np.pi/6 - eps, 2*np.pi/3 + eps, -np.pi/6 + eps, -2*np.pi/3 - eps
    state.SetFromVector((start1,start2,-start1,-start2,np.pi+start1,start2,np.pi-start1,-start2, 1, 1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.))  # (theta1, theta2, theta1dot, theta2dot)

    # Simulate for 10 seconds
    simulator.StepTo(time)
    #import pdb; pdb.set_trace()
    return (logger.data()[8:11, :], logger.data()[:8, :], logger.data()[19:22, :], logger.data()[11:19, :], logger.sample_times())

def playbackMotion(data1, data2, data3, data4, times):
    data = np.concatenate((data2, data1, data4, data3), axis=0)
    tree = RigidBodyTree(FindResource(os.path.dirname(os.path.realpath(__file__)) + "/block_pusher2.urdf"),
                    FloatingBaseType.kFixed)

    # Set up a block diagram with the robot (dynamics), the controller, and a
    # visualization block.
    builder = DiagramBuilder()
    robot = builder.AddSystem(Player(data, times))


    Tview = np.array([[1., 0., 0., 0.],
                          [0., 1., 0., 0.],
                          [0., 0., 0., 1.]],
                         dtype=np.float64)
    visualizer = builder.AddSystem(PlanarRigidBodyVisualizer(tree,Tview,
                                                             xlim=[-2.8, 4.8],
                                                             ylim=[-2.8, 10]))
    print(robot.get_output_port(0).size())
    builder.Connect(robot.get_output_port(0), visualizer.get_input_port(0))


    logger = builder.AddSystem(SignalLogger(tree.get_num_positions() + tree.get_num_velocities()))
    builder.Connect(robot.get_output_port(0), logger.get_input_port(0))

    diagram = builder.Build()

    # Set up a simulator to run this diagram
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(True)


    # Simulate for 10 seconds
    simulator.StepTo(times[-1] + 2)

def getPredictedMotion(B, v_command, object_positions, manipulator_positions, object_velocities, manipulator_velocities, times):
    object_positions = object_positions + 0.1
    manipulator_positions = manipulator_positions + 0.1
    object_velocities = object_velocities + 0.1
    manipulator_velocities = manipulator_velocities + 0.1

    object_positions = object_positions[:, range(0,object_positions.shape[1],2)]
    manipulator_positions = manipulator_positions[:, range(0,manipulator_positions.shape[1],2)]
    object_velocities = object_velocities[:, range(0,object_velocities.shape[1],2)]
    manipulator_velocities = manipulator_velocities[:, range(0,manipulator_velocities.shape[1],2)]
    times = times[range(0,times.size,2)]
    return (object_positions, manipulator_positions, object_velocities, manipulator_velocities, times)