"""
Example random agent script using the gym3 API to demonstrate that procgen works
"""
import gym3
from gym3 import types_np
import numpy as np
from procgen import ProcgenGym3Env



env = ProcgenGym3Env(num=1, env_name="bossfight2", agent_health=5, use_backgrounds=False, restrict_themes=True)#, render_mode="rgb_array")
#env = gym3.ViewerWrapper(env, info_key="rgb")

T = 100
agent_health = 2
alpha = .000001 # Learning rate

action_duration = 4 # steps to repeat action

total_episodes = 0
successful_episodes = 0
cumulative_rew = 0

left_decis = 0




w = np.random.randn(1,64*64*3)/(64*64*3)

X = np.zeros((64*64*3,T+1))
y = np.zeros(T+1)

step = 0


while True:
    rew, obs, first = env.observe()
    cumulative_rew += rew

    if step > 0 and first: # First step of new episode
        step = 0
        
        total_episodes += 1
        if cumulative_rew > -agent_health:
            successful_episodes += 1 # Successful episode!

            # REINFORCE update
            w = w + alpha * np.mean(y*X, axis=1).T
            #w = w - np.mean(w)
        cumulative_rew = 0
        print(f"{successful_episodes} of {total_episodes} successful ({successful_episodes/(total_episodes+1):.2f}%). |w| {np.linalg.norm(w):.2f} <w> {np.mean(w):.3f} frac left {left_decis/(total_episodes+1):.2f}")

    x = obs['rgb'].flatten()

    X[:,step] = x

    a = np.sign(w @ x)
    if total_episodes < 1000:
    #if np.random.rand() < 0.1: # epsilon greedy
        a = np.random.rand()-1/2

    if a > 0:
        env.act(np.array([7])) # Left
        left_decis += 1
        y[step] = 1
    else:
        env.act(np.array([7])) # Right
        y[step] = -1

   
    
    
    

    #print(obs['rgb'].shape)
    #print(types_np.sample(env.ac_space, bshape=(env.num,)))


    
    step += 1
