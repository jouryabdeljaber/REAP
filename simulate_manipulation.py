
import matplotlib.pyplot as plt
import numpy as np
#import lemkelcp as lcp
#import lcp
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







class QuasiStaticRigidBodyPlant(VectorSystem):
    """ Defines a feedback controller for the double pendulum.

    The controller applies torques at the joints in order to
    1) cancel out the dynamics of the double pendulum,
    2) make the first joint swing with the dynamics of a single pendulum, and
    3) drive the second joint towards zero.

    The magnitude of gravity for the imposed single pendulum dynamics is taken
    as a constructor argument.  So you can do fun things like pretending that
    gravity is zero, or even inverting gravity!

    """

    def __init__(self, rbt, time_step, A, B):#, rigid_body_tree, gravity):
        # 4 inputs (double pend state), 2 torque outputs.
        self.num_positions = rbt.get_num_positions()
        self.num_velocities = rbt.get_num_velocities()
        self.num_cmds = rbt.get_num_actuators() - 3
        VectorSystem.__init__(self, self.num_cmds, self.num_positions + self.num_velocities)
        VectorSystem._DeclarePeriodicDiscreteUpdate(self, time_step)
        VectorSystem._DeclareDiscreteState(self, self.num_positions + self.num_velocities)
        self.tree = rbt
        #self.g = gravity
        self.time_step = time_step
        self.A = A
        self.B = B


    def _DoCalcVectorDiscreteVariableUpdates(self, context, vel_cmd, state, next_state):
        #print(np.size(dx))
        qm = state[:self.num_positions]
        vm = state[-self.num_velocities:]
        um = 0. * vel_cmd
        for i in range(0,4):
            #import pdb; pdb.set_trace()
            um[(2*i):(2*i+2)] = IVK(qm[(2*i):(2*i+2)],vel_cmd[(2*i):(2*i+2)])
        cache = self.tree.doKinematics(qm,0.0*vm)
        (phi, JN, JT, n_con, n_tan) = self.contactGeometry(cache)

        DIM = n_con*(2 + n_tan)
        #import pdb; pdb.set_trace()
        JNO = JN[:,(-3):]
        JTO = JT[:,(-3):]
        JNM = JN[:,:(-3)]
        JTM = JT[:,:(-3)]
        JO = np.concatenate((JTO, JNO), axis=0)
        JM = np.concatenate((JTM, JNM), axis=0)
        h = self.time_step
        M = np.zeros((DIM,DIM))
        q = np.zeros(DIM)
        M[0:(n_con*(n_tan+1)), 0:(n_con*(n_tan+1))] = JO.dot( self.A.dot(JO.transpose()) ) + JM.dot( self.B.dot(JM.transpose()) )
        M[0:(n_con*n_tan),-n_con:] = np.repeat(np.eye(n_con), n_tan, axis=0)
        M[-n_con:,0:(n_con*n_tan)] = -np.repeat(np.eye(n_con), n_tan, axis=1)


        q[0:(n_con*(n_tan+1))] = JM.dot(self.time_step * um)
        q[(n_con*n_tan):(n_con*(n_tan+1))] += phi
        
        z = lcp.solveLCP(M,q)
        #print(z)
        #import pdb; pdb.set_trace()
        #if exit_code is not 0:
        #    import pdb; pdb.set_trace()
        #    raise Exception('Solution not found!')
        qp = qm
        vp = vm

        vp[-3:] = self.A.dot( JO.transpose().dot( z[:(n_con*(n_tan+1))] ) )/self.time_step
        vp[:-3] = self.B.dot( JM.transpose().dot( z[:(n_con*(n_tan+1))] ) )/self.time_step + um

        qp[-3:] += self.A.dot( JO.transpose().dot( z[:(n_con*(n_tan+1))] ) )
        qp[:-3] += self.B.dot( JM.transpose().dot( z[:(n_con*(n_tan+1))] ) ) + self.time_step * um
        next_state[:] = np.concatenate((qp, vp), axis=0)
        

    def contactGeometry(self, cache):

        (phi, normal, xA, xB, idxA, idxB) = self.tree.collisionDetect(cache)
        #import pdb; pdb.set_trace()
        n_con = np.size(phi)
        tangents = self.tree.surfaceTangents(normal)
        #import pdb; pdb.set_trace()
        tangents2 = [0] * 2 * len(tangents)
        for j in range(0, len(tangents)):
            tangents2[j] = tangents[j]
            tangents2[j + len(tangents)] = -tangents[j]
        tangents = tangents2


        n_tan = len(tangents)
        JN = np.zeros((n_con,self.num_velocities))
        JT = np.zeros((n_tan*n_con,self.num_velocities))
        #import pdb; pdb.set_trace()
        for i in range(0, n_con):
            JA = self.tree.transformPointsJacobian(cache, xA[:,i], idxA[i], 0, False)
            JB = self.tree.transformPointsJacobian(cache, xB[:,i], idxB[i], 0, False)
            JN[i,:] = normal[:,i].dot(JA - JB)
            for j in range(0,n_tan):
                                JT[i*n_tan + j,:] = tangents[j][:,i].dot(JA - JB)
        return (phi, JN, JT, n_con, n_tan)
        #self.tree.collisionDetect(cache)#, phi, normal, xA, xB, idxA, idxB)


    def _DoCalcVectorOutput(self, context, torque, state, output):
        output[:] = state






class QController(VectorSystem):

    def __init__(self, rigid_body_tree, v_des, time_step):
        self.tree = rigid_body_tree
        self.v_des = v_des
        VectorSystem.__init__(self, 0, self.tree.get_num_actuators() - 3)

    def _DoCalcVectorOutput(self, context, man_obj_state, ctrl_state, vel_cmd):
        vel_cmd[:] = self.v_des(context.get_time())







class DController(VectorSystem):

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
    #print(robot.get_output_port(0).size())
    builder.Connect(robot.get_output_port(0), visualizer.get_input_port(0))


    logger = builder.AddSystem(SignalLogger(tree.get_num_positions() + tree.get_num_velocities()))
    builder.Connect(robot.get_output_port(0), logger.get_input_port(0))

    diagram = builder.Build()

    # Set up a simulator to run this diagram
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(True)


    # Simulate for 10 seconds
    simulator.StepTo(times[-1] + 0.5)

def getPredictedMotion(B, v_command, object_positions, manipulator_positions, object_velocities, manipulator_velocities, times):
    #object_positions = object_positions + 0.1
    #manipulator_positions = manipulator_positions + 0.1
    #object_velocities = object_velocities + 0.1
    #manipulator_velocities = manipulator_velocities + 0.1

    #object_positions = object_positions[:, range(0,object_positions.shape[1],2)]
    #manipulator_positions = manipulator_positions[:, range(0,manipulator_positions.shape[1],2)]
    #object_velocities = object_velocities[:, range(0,object_velocities.shape[1],2)]
    #manipulator_velocities = manipulator_velocities[:, range(0,manipulator_velocities.shape[1],2)]
    #times = times[range(0,times.size,2)]
    #import pdb; pdb.set_trace()
    time = times[-1]
    step = 0.01
    A = 10*np.eye(3)

    tree = RigidBodyTree(FindResource(os.path.dirname(os.path.realpath(__file__)) + "/block_pusher2.urdf"),
                    FloatingBaseType.kFixed)

    # Set up a block diagram with the robot (dynamics), the controller, and a
    # visualization block.
    builder = DiagramBuilder()
    #robot = builder.AddSystem(RigidBodyPlant(tree))
    robot = builder.AddSystem(QuasiStaticRigidBodyPlant(tree, step, A, np.linalg.inv(B)))


    controller = builder.AddSystem(QController(tree, v_command, step))
    #builder.Connect(robot.get_output_port(0), controller.get_input_port(0))
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
    state = context.get_mutable_discrete_state_vector()
    start1 = 3*np.pi/16
    start2 = 15*np.pi/16
    #np.pi/6 - eps, 2*np.pi/3 + eps, -np.pi/6 + eps, -2*np.pi/3 - eps,    np.pi/6 - eps, 2*np.pi/3 + eps, -np.pi/6 + eps, -2*np.pi/3 - eps
    state.SetFromVector((start1,start2,-start1,-start2,np.pi+start1,start2,np.pi-start1,-start2, 1, 1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.))  # (theta1, theta2, theta1dot, theta2dot)
    # Simulate for 10 seconds
    simulator.StepTo(time)
    #import pdb; pdb.set_trace()
    return (logger.data()[8:11, :], logger.data()[:8, :], logger.data()[19:22, :], logger.data()[11:19, :], logger.sample_times())