"""
Implementation of Q-learning algorithm
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torchtyping import TensorType as TT

from gfn.containers import Trajectories
from gfn.env import Env
from gfn.gflownet.base import PFBasedGFlowNet
from gfn.modules import GFNModule
from gfn.samplers import Sampler


class QLearning(PFBasedGFlowNet[Trajectories]):
    def __init__(
        self,
        pf: GFNModule,
        pf_target: GFNModule = None,
        type: str = "ddqn",
        gamma: float = 1.0,
        tau: float = 0.01,
        n_step: int = 1,
    ):
        """
        Classic Q-Learning implementation

        Parameters
            ----------
            pf: GFNModule
                The Q(s,a) function approximator
            pf_target: GFNModule
                The target Q(s,a) function approximator. Only used if type is "ddqn" or "double"
            type: str
                The type of Q-learning algorithm to use. One of ["ddqn", "dqn"]
            gamma: float
                The discount factor
            tau: float
                The soft update factor for the target network
            n_step: int
                The number of steps to look ahead when computing the Q-learning target
        """
        super().__init__(pf=pf, pb=None, on_policy=False)

        assert type in ["ddqn", "dqn"]
        self.type_ = type
        self.gamma = gamma
        self.n_step = n_step

        if self.type_ in ["ddqn"]:
            assert pf_target is not None
            self.pf_target = pf_target
            self.tau = tau
            self.update(tau=1.0)
        else:
            self.pf_target = self.pf

    def sample_trajectories(
        self,
        env: Env,
        n_samples: int,
        epsilon: float = 0.0,
        p_greedy_sample: bool = False,
        high_q_sample: bool = False,
        p: float = 1.0,
    ) -> Trajectories:
        """
        Sample trajectories using one of many strategies

        Parameters
        ----------
        env: Env
            The environment to sample trajectories from
        n_samples: int
            The number of trajectories to sample
        epsilon: float
            The epsilon value to use for epsilon-greedy sampling

        TODO: add support for mixture sampling
        """
        sampler = Sampler(estimator=self.pf, is_greedy=True, epsilon=epsilon)
        trajectories = sampler.sample_trajectories(env, n_trajectories=n_samples)
        return trajectories

    def to_training_samples(self, trajectories: Trajectories) -> Trajectories:
        """
        We need to return full trajectories since the shared replay buffer between
        the QLearning and GFN models can only contain full trajectories

        Parameters
        ----------
        trajectories: Trajectories
            The batch of trajectories to convert to training samples
        """
        return trajectories

    def update(self, tau: float = None):
        """
        If we are using double q-learning, we need to copy the parameters of the
        model to the target model. This is called every ddqn_update steps

        Parameters
        ----------
        tau: float
            The soft update factor for the target network to override the default value
        """
        assert self.type_ in [
            "ddqn",
            "double",
        ], "update() should not be called if not using double q-learning"

        tau = tau if tau is not None else self.tau

        for tp, p in zip(self.pf_target.parameters(), self.pf.parameters()):
            tp.data.copy_(tau * p + (1 - tau) * tp)

    def loss(self, env: Env, trajectories: Trajectories) -> TT[0, float]:
        """
        Compute the losses over the trajectories contained in the batch

        Parameters
        ----------
        env: Env
            The environment to sample trajectories from
        trajectories: Trajectories
            The batch of trajectories to compute the losses over

        Returns
        -------
        loss: float
            The loss over the batch of trajectories
        """
        # fill value is the value used for invalid states (sink state usually)
        if trajectories.is_backward:
            raise ValueError("Backward trajectories are not supported")

        device = trajectories.states.device
        num_trajectories = trajectories.n_trajectories
        trajectory_lengths = trajectories.when_is_done
        rewards = torch.exp(trajectories.log_rewards)

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(num_trajectories, device=device).repeat_interleave(
            trajectory_lengths
        )
        # The first and last indices of the states in each trajectory
        first_state_idx = torch.cumsum(trajectory_lengths, dim=0) - trajectory_lengths
        last_state_idx = torch.cumsum(trajectory_lengths, dim=0) - 1

        # Convert trajectories to transitions and remove sink states
        (
            current_states,
            current_actions,
        ) = trajectories.to_linearized_states_and_actions()
        current_actions = current_actions.tensor

        # Number of states in the batch should match the number of actions
        assert current_states.batch_shape != tuple(
            trajectories.actions.batch_shape
        ), "Something wrong happening with log_pf evaluations"

        # Forward pass of the model, returns logits over the action space
        # that we will interpret as Q(s,a) estimates. We use the target network
        # for the next state to compute the Q(s',a') values
        Q_sa = self.pf(current_states)
        with torch.no_grad():
            Q_sa_target = self.pf_target(current_states)

        if self.type_ == "dqn":
            V_s = torch.max(Q_sa, dim=-1)[0]
        elif self.type_ == "ddqn":
            # Q(s, a) = r + γ * Q'(s', argmax Q(s', a'))
            # V_s : (sum(batch.traj_lens),)
            # next_actions = torch.argmax(Q_sa, dim=-1, keepdim=True)
            # V_s = Q_sa_target.gather(dim=-1, index=next_actions).squeeze(-1)
            V_s = torch.max(Q_sa_target, dim=-1)[0]
        elif self.type_ == "mellowmax":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown type {self.type_}")

        # Obtain the Q(s,a) values for the actions taken in the trajectories
        # Q_sa (old) : (sum(batch.traj_lens), |A|)
        # Q_sa (new) : (sum(batch.traj_lens),)
        Q_sa = Q_sa.gather(dim=-1, index=current_actions).squeeze(-1)

        # We now compute the Q learning target which is the shifted V(s) values. Depending
        # on the number of steps we look ahead (n_step), we process these differently
        if self.n_step == 1:
            # When n_step = 1, we can simply shift the V(s) values by 1 and use that as the target
            # We also need to fill in the last state with the reward. We cheat here because we know
            # that there is only one reward at the end of the trajectory. Further, this environment
            # does not allow illegal actions, so we don't need to worry handling the rewards for
            # illegal trajectories
            #
            # V = [1,2,3,4,5,6,7]
            # shifted_V = [1,2,R1,4,5,6,R2]

            shifted_V = self.gamma * torch.cat([V_s[1:], torch.zeros_like(V_s[:1])])
            shifted_V[last_state_idx] = rewards
            hat_Q = shifted_V
        else:
            # When n_step > 1, we need to compute the n-step return. This is done by shifting the
            # V(s) values by n_step and then merging the V(s) values with the rewards. We need to
            # be careful here because the trajectory might end before n_step steps, in which case
            # we need to use the reward instead of the V(s) value. Note that we once again cheat here
            # because we know there is only reward at the end of the trajectory.

            # The raw index of each state in the batch
            raw_idx = torch.arange(batch_idx.shape[0], device=device)
            # Compute the target index which is the raw index n steps into the future
            target_idx = raw_idx - first_state_idx[batch_idx] + self.n_step
            # Now, to decide whether the target is V(s_{t+n}) or R(\tau), we need to check whether
            # the idx of the target state is less than the length of the active trajectory
            mask = target_idx < trajectory_lengths[batch_idx]
            # Set the target to the estimated state value V(s_{t+n}) if the trajectory is
            # still active, otherwise set it to the reward R(\tau) at the end of the trajectory
            hat_Q = V_s.roll(-self.n_step) * mask + (rewards[batch_idx]) * (~mask)

        # Compute the Q-learning loss using huber norm
        losses = F.huber_loss(Q_sa, hat_Q, reduction="none")
        loss = losses.mean()

        if torch.isnan(loss):
            raise ValueError("loss is nan")

        info = {
            "loss": loss,
            "Q_sa": Q_sa.mean().item(),
        }

        return loss, info

    # Deprecated
    def old_loss(self, env: Env, trajectories: Trajectories) -> TT[0, float]:
        # fill value is the value used for invalid states (sink state usually)
        if trajectories.is_backward:
            raise ValueError("Backward trajectories are not supported")

        transitions = trajectories.to_transitions()
        nmask = ~transitions.next_states.is_sink_state

        log_rewards = transitions.log_rewards
        is_done = transitions.is_done
        actions = transitions.actions

        if log_rewards is None:
            raise ValueError("log_rewards is None")

        states = transitions.states
        next_states = transitions.next_states[nmask]

        max_next_Q = torch.zeros_like(log_rewards)

        if states.batch_shape != tuple(actions.batch_shape):
            raise ValueError("Something wrong happening with log_pf evaluations")

        curr_Q = self.pf(states)

        with torch.no_grad():
            next_Q = self.pf_target(next_states)

        if next_Q.shape[0] != 0:
            max_next_Q[nmask] = torch.max(next_Q, dim=-1)[0]

        current_Q = curr_Q.gather(dim=-1, index=actions.tensor).squeeze(-1)
        expected_Q = torch.exp(log_rewards) + self.gamma * max_next_Q

        max_traj_length = trajectories.when_is_done.float().max()
        avg_traj_length = trajectories.when_is_done.float().mean()
        max_reward = torch.exp(log_rewards[is_done]).max()
        avg_reward = torch.exp(log_rewards[is_done]).mean()

        assert is_done.int().sum() == trajectories.n_trajectories

        loss = F.mse_loss(current_Q, expected_Q)

        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss, avg_traj_length, avg_reward, max_traj_length, max_reward
