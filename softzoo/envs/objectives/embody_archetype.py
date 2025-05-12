import torch
import taichi as ti

from .base import Base


class EmbodyArchetype(Base):
    def __init__(self, env, config):
        super().__init__(env, config)

        self.max_episode_steps = self.config.get('max_episode_steps', self.env.max_steps)
        assert self.max_episode_steps <= self.env.max_steps
        assert self.config['max_episode_steps'] != torch.inf, 'Maximal episode step is infinite'

        offset_des, amplitude_des, period_des = 0.069, 0.030, 0.3
        self.offset_des = offset_des
        self.amplitude_des = amplitude_des
        self.period_des = period_des

    def reset(self):
        self.step_cnt = 0

    def get_obs(self, s):
        x, mask = self.env.design_space.get_x(s,keep_mask=True) 
        p_min = self.env.design_space.orientation_data['min_p']
        p_max = self.env.design_space.orientation_data['max_p']
        obs = torch.stack([x[p_min, :], x[p_max, :]], dim=0)
        return obs

    def get_reward(self, s):
        self.step_cnt += 1
        
        # get observation
        obs = self.get_obs(s)
        # get time
        time = torch.tensor(self.env.sim.solver.sim_t, dtype=obs.dtype, device=obs.device)

        # relative distance between the two particles
        rel_dist = obs[-1, :] - obs[0, :]
        rel_dist_norm = torch.norm(rel_dist, dim=0)
        target = self.offset_des + self.amplitude_des * torch.cos(time * 2 * torch.pi / self.period_des)
        loss = 1e1 * torch.square(rel_dist_norm - target)

        # print('embody archetype loss:', loss)
        rew = -loss.item()

        return rew

    def get_done(self):
        return not (self.step_cnt < self.max_episode_steps)

    @property
    def obs_shape(self):
        return (2, 3)
