# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from eval import evaluate_agent
from utils import make_env
from tasks import TaskManager
from agent import BasePPOAgent, ParsevalPPOAgent
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # PPO specific arguments computed
    algorithm: str = "base"
    """the type of PPO algorithm: base, parseval"""
    parseval_reg: float = 0.0001
    """the Parseval regularization strength (should try both 0.001 & 0.0001)"""
    net_width: int = 64
    """Width of the network layers"""
    add_diag_layer: bool = True
    """Whether to add a diagonal layer in the Parseval network"""
    activation: str = 'tanh'
    """Activation function to use in the Parseval network"""
    input_scale: float = 1
    """Input scaling factor for the Parseval network"""
    learnable_input_scale: bool = False
    """Whether the input scaling factor is learnable"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # for evaluation
    eval_episodes: int = 10
    """number of episodes to test the agent during evaluation"""
    target_return: int = 475
    """target return to consider the task solved (only for CartPole-v1)"""
    stable_hits_required: int = 10
    """number of stable hits to consider convergence"""

    # for task switching
    task_switch_episode_interval: int = 100
    """number of episodes between each task switch"""
    num_tasks: int = 4
    """total number of tasks"""

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.algorithm}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    if args.algorithm == "base":
        agent = BasePPOAgent(envs).to(device)
    elif args.algorithm == "parseval":
        agent = ParsevalPPOAgent(envs, net_width=args.net_width,
                                add_diag_layer=args.add_diag_layer,
                                activation=args.activation,
                                input_scale=args.input_scale,
                                learnable_input_scale=args.learnable_input_scale).to(device)
    task_manager = TaskManager(envs)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # additional variables
    episode_count = 0
    convergence_episode = None
    best_task_performance = {t: -np.inf for t in range(args.num_tasks)}
    performance_matrix = np.zeros((args.num_tasks, args.num_tasks))
    task_stable_hits = {t: 0 for t in range(args.num_tasks)}

    # Start with Task 1
    current_task = 0
    task_manager.set_task(current_task)
    writer.add_scalar("charts/task_id", current_task, global_step)
    best_per_task = {t: -np.inf for t in range(args.num_tasks)}
    task_seen = [False] * args.num_tasks

    if args.env_id == "CartPole-v1":
        success_threshold = 500

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            current_cycle = episode_count // args.task_switch_episode_interval
            task_seen[current_task] = True

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episode_count += 1
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], episode_count)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], episode_count)
                        writer.add_scalar("charts/global_step_return", info["episode"]["r"], global_step)
                        
                        if episode_count % args.task_switch_episode_interval == 0:
                            prev_task = current_task
                            current_task = (current_task + 1) % args.num_tasks
                            task_manager.set_task(current_task)
                            print(f"---- SWITCH TO TASK {current_task + 1} ----")

                            # ==================================
                            # 2. RUN EVALUATION ON *ALL* TASKS
                            # ==================================
                            task_returns = []
                            task_success = []
                            forgetting_scores = []
                            for task_id in range(args.num_tasks):
                                mean_return, success_rate = evaluate_agent(
                                    agent, make_env, args, device,
                                    success_threshold=success_threshold,
                                    task_manager=task_manager,
                                    task_id=task_id,
                                    eval_episodes=args.eval_episodes
                                )

                                # Log results
                                writer.add_scalar(f"eval/task_{task_id}/mean_return", mean_return, episode_count)
                                writer.add_scalar(f"eval/task_{task_id}/success_rate", success_rate, episode_count)
                                task_returns.append(mean_return)
                                task_success.append(success_rate)

                                # Update performance matrix
                                performance_matrix[task_id][prev_task] = mean_return

                                # Update best performance
                                best_per_task[task_id] = max(best_per_task[task_id], mean_return)

                                # compute forgetting only if:
                                # (1) the task has been seen before
                                # (2) we are evaluating after first cycle (episode ≥ full cycle)
                                if task_seen[task_id] and current_cycle > 0:
                                    forgetting = best_per_task[task_id] - mean_return
                                else:
                                    forgetting = 0.0  # undefined in CL; treat as zero   
                                writer.add_scalar(f"eval/task_{task_id}/forgetting", forgetting, episode_count)
                                forgetting_scores.append(forgetting)

                                # ---- per-task convergence tracking ----
                                if success_rate == 1.0:
                                    task_stable_hits[task_id] += 1
                                else:
                                    task_stable_hits[task_id] = 0   # reset streak
                                
                                # check if all tasks have converged
                                if all([task_stable_hits[t] >= args.stable_hits_required for t in range(args.num_tasks)]):
                                    if convergence_episode is None:
                                        convergence_episode = episode_count
                                        writer.add_scalar("eval/convergence_episode", convergence_episode, episode_count)
                                        writer.add_text("eval/convergence_info", f"Converged at episode {convergence_episode}", episode_count)
                                        print(f"[CONVERGED] at episode {convergence_episode}")
                            writer.add_scalar("eval/avg_mean_return", np.mean(task_returns), episode_count)
                            writer.add_scalar("eval/avg_success_rate", np.mean(task_success), episode_count)
                            writer.add_scalar("eval/avg_forgetting", np.mean(forgetting_scores), episode_count)


            writer.add_scalar("charts/task_id_episode", current_task, episode_count)

        # ------------------------------------
        # _update_parameters (start)
        # ------------------------------------
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # ------------------------------------
                #  Parseval regularization (start)
                # ------------------------------------
                if args.algorithm == "parseval":
                    loss = loss + args.parseval_reg * (64 / args.net_width)**2 * agent.parseval_reg_network(agent.actor.named_parameters())
                    loss = loss + args.parseval_reg * (64 / args.net_width)**2 * agent.parseval_reg_network(agent.critic.named_parameters())
                # ------------------------------------
                #  Parseval regularization (end)
                # ------------------------------------
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        # ------------------------------------
        # _update_parameters (end)
        # ------------------------------------
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("charts/task_id", current_task, global_step)
            
    envs.close()
    writer.close()
