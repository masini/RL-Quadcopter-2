import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * self.sim.pose.size
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        if target_pos is None :
            print("Setting default init pose")
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10., 0., 0., 0.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""

        # reward for Z is calculated apart to give it more importance
        reward_z = 1-np.tanh(abs(self.sim.pose[2]-self.target_pos[2])/self.target_pos[2])
        reward_z *= 3

        # reward for other coordinates
        reward_all = 1-np.tanh(abs(self.sim.pose[0:2]-self.target_pos[0:2]).sum())
        reward_all += 1-np.tanh(abs(self.sim.pose[3:]-self.target_pos[3:]).sum())

        reward = reward_z + reward_all

        return reward * 10

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            new_reward = self.get_reward()
            reward = reward + new_reward if new_reward > reward else reward -1
            pose_all.append(self.sim.pose)
            if done :
                reward += 10
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state