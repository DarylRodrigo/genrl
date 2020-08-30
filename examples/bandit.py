from pathlib import Path

from matplotlib import pyplot as plt

from genrl.agents import bandits

from genrl.agents import EpsGreedyMABAgent, BernoulliMAB
from genrl.trainers import MABTrainer, DCBTrainer

bandit = BernoulliMAB(arms=50, context_type="int")  
agent = EpsGreedyMABAgent(bandit, eps=0.1)
trainer = MABTrainer(agent, bandit)             
results = trainer.train(2000)

fig, axs = plt.subplots(2, 2, figsize=(15, 12), dpi=600)
fig.suptitle("Deep Contextual Bandit Example", fontsize=14)
axs[0, 0].set_title("Cumulative Regret")
axs[0, 1].set_title("Cumulative Reward")
axs[1, 0].set_title("Regret Moving Avg")
axs[1, 1].set_title("Reward Moving Avg")
axs[0, 0].plot(results["cumulative_regrets"], label="neural-linpos")
axs[0, 1].plot(results["cumulative_rewards"], label="neural-linpos")
axs[1, 0].plot(results["regret_moving_avgs"], label="neural-linpos")
axs[1, 1].plot(results["reward_moving_avgs"], label="neural-linpos")

plt.legend()
fig.savefig(Path("logs/").joinpath("dcb_example.png"))
