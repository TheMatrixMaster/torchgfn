r"""
Implement the baseline RL algorithm for the hypergrid environment.

Example usage:
python run.py --ndim 4 --height 8 --R0 0.01 --loss TD --type ddqn
"""

import os
from argparse import ArgumentParser

import torch
import wandb
from tqdm import tqdm, trange

from gfn.containers import ReplayBuffer
from gfn.gflownet import QLearning
from gfn.gym import HyperGrid
from gfn.modules import DiscretePolicyEstimator
from gfn.utils.common import get_metrics, validate
from gfn.utils.modules import NeuralNet


def main(args):  # noqa: C901
    seed = args.seed if args.seed != 0 else torch.randint(int(10e10), (1,))[0].item()
    torch.manual_seed(seed)

    device_str = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    use_wandb = len(args.wandb_project) > 0
    if use_wandb:
        offline = os.environ.get("WANDB_MODE", "offline")
        wandb.init(
            project=args.wandb_project,
            dir=f"{args.log_dir}",
            resume="allow",
            mode=offline,
        )
        wandb.config.update(args)

    # 1. Create the environment
    env = HyperGrid(
        args.ndim, args.height, args.R0, args.R1, args.R2, device_str=device_str
    )

    # 2. Create the double dqn qlearning model.
    pf_module = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
    )
    pf_target_module = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
    )

    assert pf_module is not None, f"pf_module is None. Command-line arguments: {args}"
    assert (
        pf_target_module is not None
    ), f"pf_target_module is None. Command-line arguments: {args}"

    pf_estimator = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
        preprocessor=env.preprocessor,
    )
    pf_target_estimator = DiscretePolicyEstimator(
        module=pf_target_module,
        n_actions=env.n_actions,
        preprocessor=env.preprocessor,
    )

    qlearner = QLearning(
        pf=pf_estimator,
        pf_target=pf_target_estimator,
        type=args.ql_type,
        gamma=args.gamma,
        tau=args.tau,
        n_step=args.n_step,
    )

    assert qlearner is not None, "No qlearner for ddqn"

    # Initialize the replay buffer ?

    replay_buffer = None
    if args.replay_buffer_size > 0:
        if args.loss in ("TB", "SubTB", "ZVar"):
            objects_type = "trajectories"
        elif args.loss in ("DB", "ModifiedDB"):
            objects_type = "transitions"
        elif args.loss == "FM":
            objects_type = "states"
        else:
            raise NotImplementedError(f"Unknown loss: {args.loss}")
        replay_buffer = ReplayBuffer(
            env, objects_type=objects_type, capacity=args.replay_buffer_size
        )

    # 3. Create the optimizer

    # Policy parameters have their own LR.
    params = [
        {
            "params": [v for k, v in dict(qlearner.named_parameters()).items()],
            "lr": args.lr,
        }
    ]

    eps = args.eps
    eps_min = args.eps_min
    eps_decay = args.eps_decay
    tau_step = args.tau_step
    optimizer = torch.optim.Adam(params)

    visited_terminating_states = env.States.from_batch_shape((0,))

    states_visited = 0
    n_iterations = args.n_trajectories // args.batch_size
    validation_info = {"l1_dist": float("inf")}

    for iteration in trange(n_iterations):
        trajectories = qlearner.sample_trajectories(
            env, n_samples=args.batch_size, epsilon=eps
        )
        training_samples = qlearner.to_training_samples(trajectories)

        if replay_buffer is not None:
            with torch.no_grad():
                replay_buffer.add(training_samples)
                training_objects = replay_buffer.sample(n_trajectories=args.batch_size)
        else:
            # on policy
            training_objects = training_samples

        optimizer.zero_grad()
        loss, _ = qlearner.loss(env, training_objects)
        loss.backward()
        optimizer.step()

        if args.ql_type == "ddqn" and iteration % tau_step == 0:
            qlearner.update()

        visited_terminating_states.extend(trajectories.last_states)
        states_visited += len(trajectories)
        metrics = get_metrics(trajectories)

        to_log = {
            "eps": eps,
            "loss": loss.item(),
            "states_visited": states_visited,
            **metrics,
        }

        if args.reward_based_eps:
            if metrics["max_reward"] > (2.0 + args.R0):
                eps = max(eps * eps_decay, eps_min)
        else:
            eps = max(eps * eps_decay, eps_min)

        if use_wandb:
            wandb.log(to_log, step=iteration)
        if iteration % args.validation_interval == 0:
            validation_info = validate(
                env,
                qlearner,
                args.validation_samples,
                visited_terminating_states,
            )
            if use_wandb:
                wandb.log(validation_info, step=iteration)

            to_log.update(validation_info)
            tqdm.write(f"{iteration}: {to_log}")

    return validation_info["l1_dist"]


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")

    parser.add_argument(
        "--ndim", type=int, default=2, help="Number of dimensions in the environment"
    )
    parser.add_argument(
        "--height", type=int, default=8, help="Height of the environment"
    )
    parser.add_argument("--R0", type=float, default=0.1, help="Environment's R0")
    parser.add_argument("--R1", type=float, default=0.5, help="Environment's R1")
    parser.add_argument("--R2", type=float, default=2.0, help="Environment's R2")

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed, if 0 then a random seed is used",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size, i.e. number of trajectories to sample per training iteration",
    )
    parser.add_argument(
        "--replay_buffer_size",
        type=int,
        default=0,
        help="If zero, no replay buffer is used. Otherwise, the replay buffer is used.",
    )

    parser.add_argument(
        "--loss",
        type=str,
        choices=["TD"],
        default="TD",
        help="Loss function to use",
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension of the estimators' neural network modules.",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=2,
        help="Number of hidden layers (of size `hidden_dim`) in the estimators'"
        + " neural network modules",
    )

    parser.add_argument(
        "--ql_type",
        type=str,
        choices=["dqn", "ddqn"],
        default="ddqn",
        help="Type of Q-learning algorithm to use",
    )

    parser.add_argument(
        "--reward_based_eps",
        action="store_true",
        help="Whether to use a reward-based epsilon greedy policy",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1.0,
        help="Epsilon parameter for the epsilon-greedy policy",
    )

    parser.add_argument(
        "--eps_decay",
        type=float,
        default=0.999,
        help="Epsilon decay parameter for the epsilon-greedy policy",
    )

    parser.add_argument(
        "--eps_min",
        type=float,
        default=0.01,
        help="Minimum epsilon parameter for the epsilon-greedy policy",
    )

    parser.add_argument(
        "--n_step",
        type=int,
        default=1,
        help="Number of steps to look ahead for the n-step Q-learning algorithm",
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Tau parameter for the target network update",
    )

    parser.add_argument(
        "--tau_step",
        type=int,
        default=1,
        help="Number of steps between target network updates",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Discount factor for the Q-learning algorithm",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the estimators' modules",
    )

    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=int(1e6),
        help="Total budget of trajectories to train on. "
        + "Training iterations = n_trajectories // batch_size",
    )

    parser.add_argument(
        "--validation_interval",
        type=int,
        default=100,
        help="How often (in training steps) to validate the gflownet",
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=200000,
        help="Number of validation samples to use to evaluate the probability mass function.",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="",
        help="Name of the wandb project. If empty, don't use wandb",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="~/scratch",
        help="Directory where to log the results.",
    )

    args = parser.parse_args()

    print(main(args))
