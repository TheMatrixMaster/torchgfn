r"""
Implement the mixture policy between gflownet and ddqn.

Example usage:
python run.py --ndim 4 --height 8 --R0 0.01
"""

import os
from argparse import ArgumentParser

import torch
import wandb
from tqdm import tqdm, trange

from gfn.containers import ReplayBuffer
from gfn.gflownet import QLearning, TBGFlowNet
from gfn.gym import HyperGrid
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import MixtureSampler
from gfn.utils.common import get_metrics, validate
from gfn.utils.modules import DiscreteUniform, NeuralNet


def setup_ddqn(args, env):
    """Setup the double dqn qlearning model.

    Parameters:
        args (argparse.Namespace): Command-line arguments.
        env (gfn.gym.HyperGrid): The environment.

    Returns:
        gfn.gflownet.QLearning: The qlearning model.
        torch.optim.Adam: The optimizer for the qlearning model.
    """
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

    params = [
        {
            "params": [v for k, v in dict(qlearner.named_parameters()).items()],
            "lr": args.lr,
        }
    ]

    optimizer = torch.optim.Adam(params)

    return qlearner, optimizer


def setup_gfn(args, env):
    """Setup the gflownet model.

    Parameters:
        args (argparse.Namespace): Command-line arguments.
        env (gfn.gym.HyperGrid): The environment.

    Returns:
        gfn.gflownet.TBGFlowNet: The gflownet model.
        torch.optim.Adam: The optimizer for the gflownet model.
    """
    pf_module = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden,
    )
    if not args.uniform_pb:
        pb_module = NeuralNet(
            input_dim=env.preprocessor.output_dim,
            output_dim=env.n_actions - 1,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
            torso=pf_module.torso if args.tied else None,
        )
    else:
        pb_module = DiscreteUniform(env.n_actions - 1)

    assert pf_module is not None, f"pf_module is None. Command-line arguments: {args}"
    assert pb_module is not None, f"pb_module is None. Command-line arguments: {args}"

    pf_estimator = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
        preprocessor=env.preprocessor,
    )
    pb_estimator = DiscretePolicyEstimator(
        module=pb_module,
        n_actions=env.n_actions,
        is_backward=True,
        preprocessor=env.preprocessor,
    )

    gflownet = TBGFlowNet(
        pf=pf_estimator,
        pb=pb_estimator,
        on_policy=True if args.replay_buffer_size == 0 else False,
    )

    params = [
        {
            "params": [
                v for k, v in dict(gflownet.named_parameters()).items() if k != "logZ"
            ],
            "lr": args.lr,
        }
    ]

    # Log Z gets dedicated learning rate (typically higher).
    if "logZ" in dict(gflownet.named_parameters()):
        params.append(
            {
                "params": [dict(gflownet.named_parameters())["logZ"]],
                "lr": args.lr_Z,
            }
        )

    optimizer = torch.optim.Adam(params)

    return gflownet, optimizer


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

    # 2. Create the gflownet model.
    gflownet, gfn_optimizer = setup_gfn(args, env)

    # 3. Create the double dqn qlearning model.
    qlearner, ql_optimizer = setup_ddqn(args, env)

    assert qlearner is not None, f"Failed to setup qlearner with args: {args}"
    assert gflownet is not None, f"Failed to setup gflownet with args: {args}"

    # 4. Initialize the replay buffer ? (only support trajectories for now)

    replay_buffer = None
    objects_type = "trajectories"

    if args.replay_buffer_size > 0:
        replay_buffer = ReplayBuffer(
            env, objects_type=objects_type, capacity=args.replay_buffer_size
        )

    # 5. Setup the training loop

    eps = args.eps
    eps_min = args.eps_min
    eps_decay = args.eps_decay
    tau_step = args.tau_step

    visited_terminating_states = env.States.from_batch_shape((0,))

    states_visited = 0
    n_iterations = args.n_trajectories // args.batch_size
    validation_info = {"l1_dist": float("inf")}

    estimators = [gflownet.pf, qlearner.pf]
    mixture_flavor = args.mixture_flavor

    if mixture_flavor == "p_greedy":
        assert args.p_greedy >= 0 and args.p_greedy <= 1, "p_greedy must be in [0, 1]"
        weights = torch.tensor([args.p_greedy, 1 - args.p_greedy], device=device_str)
    elif mixture_flavor == "high_q":
        raise NotImplementedError("high_q mixture flavor not implemented yet")
    else:
        raise NotImplementedError(f"Unknown mixture flavor: {mixture_flavor}")

    for iteration in trange(n_iterations):
        # Sample trajectories using mixture sampler
        sampler = MixtureSampler(
            estimators=estimators,
            weights=weights,
            mixture_flavor=mixture_flavor,
            epsilon=eps,
        )
        trajectories = sampler.sample_trajectories(env, n_trajectories=args.batch_size)
        training_samples = trajectories  # we only support trajectories for now

        if replay_buffer is not None:
            with torch.no_grad():
                replay_buffer.add(training_samples)
                training_objects = replay_buffer.sample(n_trajectories=args.batch_size)
        else:
            # on policy
            training_objects = training_samples

        # Update the models
        gfn_optimizer.zero_grad()
        gfn_loss = gflownet.loss(env, training_objects)
        gfn_loss.backward()
        gfn_optimizer.step()

        ql_optimizer.zero_grad()
        ql_loss, _ = qlearner.loss(env, training_objects)
        ql_loss.backward()
        ql_optimizer.step()

        if args.ql_type == "ddqn" and iteration % tau_step == 0:
            qlearner.update()

        visited_terminating_states.extend(trajectories.last_states)
        states_visited += len(trajectories)
        metrics = get_metrics(trajectories)

        to_log = {
            "eps": eps,
            "gfn_loss": gfn_loss.item(),
            "ql_loss": ql_loss.item(),
            "states_visited": states_visited,
            **metrics,
        }

        if args.reward_based_eps:
            if metrics["max_reward"] > args.R2:
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

    parser.add_argument("--uniform_pb", action="store_true", help="Use a uniform PB")
    parser.add_argument(
        "--tied", action="store_true", help="Tie the parameters of PF, PB, and F"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the estimators' modules",
    )
    parser.add_argument(
        "--lr_Z",
        type=float,
        default=0.1,
        help="Specific learning rate for Z (only used for TB loss)",
    )

    parser.add_argument(
        "--p_greedy",
        type=float,
        default=0.99,
        help="Greedy factor weight when mixing between gflownet and ddqn policies",
    )

    parser.add_argument(
        "--mixture_flavor",
        type=str,
        choices=["p_greedy", "high_q"],
        default="p_greedy",
        help="How to mix between gflownet and ddqn policies",
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
