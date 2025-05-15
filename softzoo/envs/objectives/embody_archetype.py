import torch
import taichi as ti

from .base import Base


class EmbodyArchetype(Base):
    """
    Attention: The reward functions doesn't seem to be used during optimization. Define a loss function instead.
    """
    def __init__(self, env, config):
        super().__init__(env, config)

        self.max_episode_steps = self.config.get('max_episode_steps', self.env.max_steps)
        assert self.max_episode_steps <= self.env.max_steps
        assert self.config['max_episode_steps'] != torch.inf, 'Maximal episode step is infinite'

        self.offset_des = config["offset_des"]
        self.amplitude_des = config["amplitude_des"]
        self.period_des = config["period_des"]

    def reset(self):
        self.step_cnt = 0

    def get_obs(self, s):
        # get time
        time = torch.tensor(self.env.sim.solver.sim_t)

        p_min = self.env.design_space.orientation_data['min_p']
        p_max = self.env.design_space.orientation_data['max_p']

        x, mask = self.env.design_space.get_x(s, keep_mask=True)
        x_min, x_max = x[p_min, :], x[p_max, :]

        # s_local = self.env.sim.solver.get_cyclic_s(s)
        # p_start = self.env.design_space.p_start
        # x_min_vector = self.env.sim.solver.x[s_local, p_start + p_min]
        # x_max_vector = self.env.sim.solver.x[s_local, p_start + p_max]
        # x_min = torch.tensor(x_min_vector)
        # x_max = torch.tensor(x_max_vector)

        # compute the target
        target = self.offset_des + self.amplitude_des * torch.cos(time * 2 * torch.pi / self.period_des)
        # repeat three times
        target = target[None].repeat(3)

        obs = torch.stack([x_min, x_max, target], dim=0)
        return obs

    def get_reward(self, s):
        self.step_cnt += 1
        
        # get observation
        obs = self.get_obs(s)
        # extract the parts
        x_min = obs[0, :]
        x_max = obs[1, :]
        target = obs[2, 0]

        # relative distance between the two particles
        rel_dist = x_max - x_min
        rel_dist_norm = torch.norm(rel_dist, dim=0)
        loss = torch.square(rel_dist_norm - target)

        rew = -loss.item()

        return rew

    def get_done(self):
        return not (self.step_cnt < self.max_episode_steps)

    @property
    def obs_shape(self):
        return (2, 3)
