from parallelEnv import parallelEnv
# import matplotlib.pyplot as plt
import torch
import numpy as np
# from JSAnimation.IPython_display import display_animation
# from matplotlib import animation
import random as rand

RIGHT = 4
LEFT = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# preprocess a single frame
# crop image and downsample to 80x80
# stack two frames together as input
def preprocess_single(image, bkg_color=np.array([144, 72, 17])):
    img = np.mean(image[34:-16:2, ::2] - bkg_color, axis=-1) / 255.
    return img


# convert outputs of parallelEnv to inputs to pytorch neural net
# this is useful for batch processing especially on the GPU
def preprocess_batch(images, bkg_color=np.array([144, 72, 17])):
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 5:
        list_of_images = np.expand_dims(list_of_images, 1)
    # subtract bkg and crop
    list_of_images_prepro = np.mean(list_of_images[:, :, 34:-16:2, ::2] - bkg_color,
                                    axis=-1) / 255.
    batch_input = np.swapaxes(list_of_images_prepro, 0, 1)
    return torch.from_numpy(batch_input).float().to(device)


# function to animate a list of frames
def animate_frames(frames):
    plt.axis('off')

    # color option for plotting
    # use Greys for greyscale
    cmap = None if len(frames[0].shape) == 3 else 'Greys'
    patch = plt.imshow(frames[0], cmap=cmap)

    fanim = animation.FuncAnimation(plt.gcf(), \
                                    lambda x: patch.set_data(frames[x]), frames=len(frames), interval=30)

    display(display_animation(fanim, default_mode='once'))


# play a game and display the animation
# nrand = number of random steps before using the policy
def play(env, policy, time=2000, preprocess=None, nrand=5):
    env.reset()

    # star game
    env.step(1)

    # perform nrand random steps in the beginning
    for _ in range(nrand):
        frame1, reward1, is_done, _ = env.step(np.random.choice([RIGHT, LEFT]))
        frame2, reward2, is_done, _ = env.step(0)

    anim_frames = []

    for _ in range(time):

        frame_input = preprocess_batch([frame1, frame2])
        prob = policy(frame_input)

        # RIGHT = 4, LEFT = 5
        action = RIGHT if rand.random() < prob else LEFT
        frame1, _, is_done, _ = env.step(action)
        frame2, _, is_done, _ = env.step(0)

        if preprocess is None:
            anim_frames.append(frame1)
        else:
            anim_frames.append(preprocess(frame1))

        if is_done:
            break

    env.close()

    animate_frames(anim_frames)
    return


# collect trajectories for a parallelized parallelEnv object
def collect_trajectories(envs, policy, R, ratio, randrew = False, tmax=200, nrand=5, preagents=None):
    # number of parallel instances
    n = len(envs.ps)

    # initialize returning lists and start the game!
    state_list = []
    reward_list = []
    rewards = np.zeros((tmax, n))
    prob_list = []
    action_list = []
    rewards_mask = np.ones(n)
    time_od = np.zeros(n)

    # start all parallel agents
    envs.reset()
    envs.step([1] * n)

    # perform nrand random steps
    for _ in range(nrand):
        fr1, re1, _, _ = envs.step(np.random.choice([RIGHT, LEFT], n))
        fr2, re2, _, _ = envs.step([0] * n)

    if preagents is not None:
        #take random number of steps using a pretrained agent
        #choose random agent
        index = rand.randint(0,6)
        pre_steps = rand.randint(28,55)
        for _ in range(pre_steps):
            frame_input = preprocess_batch([fr1, fr2])
            probabilities = preagents[index](frame_input).squeeze().cpu().detach().numpy()
            actions = np.where(np.random.rand(n) < probabilities, RIGHT, LEFT)

            fr1, re1, is_done, _ = envs.step(actions)
            fr2, re2, is_done, _ = envs.step([0] * n)

    # for t in range(tmax):
    for t in range(tmax):
        # prepare the input
        # preprocess_batch properly converts two frames into
        # shape (n, 2, 80, 80), the proper input for the policy
        # this is required when building CNN with pytorch
        batch_input = preprocess_batch([fr1, fr2])

        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        probs = policy(batch_input).squeeze().cpu().detach().numpy()

        action = np.where(np.random.rand(n) < probs, RIGHT, LEFT)
        probs = np.where(action == RIGHT, probs, 1.0 - probs)

        # advance the game (0=no action)
        # we take one action and skip game forward
        fr1, re1, is_done, _ = envs.step(action)
        fr2, re2, is_done, _ = envs.step([0] * n)

        reward = re1 + re2

        mask = np.where(reward < 0, 0, 1)

        edited_reward = rewards_mask * (mask - 1) * ratio * R
        rewards_mask *= mask
        rewards[t, :] = np.copy(edited_reward)
        time_od += rewards_mask

        # store the result
        state_list.append(batch_input)
        prob_list.append(probs)
        action_list.append(action)

        # stop if any of the trajectories is done
        # we want all the lists to be rectangular
        if is_done.any():
            print('Done!')
            break

    # return pi_theta, states, actions, rewards, probability

    """
    we now convert the reward list to numpy array, then perform masking procedure, where any episodes with negative
    reward have all their rewards set to 0
    we give all other episodes (survived ones) rewards of zero throughout episode, except for very last timestep,
    where rewards of +R is given 
    process can be simplified by just creating this array from the reward mask, we will do that for now
    """
    # reward_list = np.asarray(reward_list)
    # rewards = np.zeros((len(action_list),n))

    #set a semirandom time to receive reward for the surviving agents
    if randrew:
        time_indices = tmax - 1 - np.random.randint(0, 12, n)
        env_indices = np.arange(n)
        rewards[time_indices, env_indices] = rewards_mask * R
    else:
        rewards[-1, :] = rewards_mask * R

    #
    return prob_list, state_list, \
           action_list, rewards, rewards_mask, time_od, fr1, fr2


# convert states to probability, passing through the policy
def states_to_prob(policy, states):
    states = torch.stack(states)
    policy_input = states.view(-1, *states.shape[-3:])
    return policy(policy_input).view(states.shape[:-3])


# return sum of log-prob divided by T
# same thing as -policy_loss
def surrogate(policy, old_probs, states, actions, rewards,
              discount=0.995, beta=0.01):
    # discount = discount ** np.arange(len(rewards))
    discount = discount ** np.arange(len(actions))

    # rewards = np.asarray(rewards) * discount[:, np.newaxis]
    rewards = rewards * discount[:, np.newaxis]
    """
    If there are any -1 in the reward list for an episode, we set all rewards in said episode to be 0
    Otherwise, we set all rewards in episode to be 0, except for the very last reward, which we set to be 1 or 2?
    """

    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:, np.newaxis]) / std[:, np.newaxis]

    # convert everything into pytorch tensors and move to gpu if available
    actions = np.asarray(actions)
    actions = torch.from_numpy(actions)
    actions = actions.to(torch.int8)
    actions = actions.to(device)

    old_probs = np.asarray(old_probs)
    old_probs = torch.from_numpy(old_probs)
    old_probs = old_probs.to(torch.float)
    old_probs = old_probs.to(device)

    rewards = np.asarray(rewards_normalized)
    rewards = torch.from_numpy(rewards)
    rewards = rewards.to(torch.float)
    rewards = rewards.to(device)

    # convert states to policy (or probability)
    new_probs = states_to_prob(policy, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0 - new_probs)

    ratio = new_probs / old_probs

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs * torch.log(old_probs + 1.e-10) + \
                (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

    return torch.mean(ratio * rewards + beta * entropy)


# clipped surrogate function
# similar as -policy_loss for REINFORCE, but for PPO
def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      discount=0.995,
                      epsilon=0.1, beta=0.01):
    discount = discount ** np.arange(len(rewards))
    rewards = np.asarray(rewards) * discount[:, np.newaxis]

    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:, np.newaxis]) / std[:, np.newaxis]

    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_probs = states_to_prob(policy, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0 - new_probs)

    # ratio for clipping
    ratio = new_probs / old_probs

    # clipped function
    clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    clipped_surrogate = torch.min(ratio * rewards, clip * rewards)

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs * torch.log(old_probs + 1.e-10) + \
                (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

    # this returns an average of all the entries of the tensor
    # effective computing L_sur^clip / T
    # averaged over time-step and number of trajectories
    # this is desirable because we have normalized our rewards
    return torch.mean(clipped_surrogate + beta * entropy)


import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    """
    def __init__(self):
        super(Policy, self).__init__()

        # 80x80 to outputsize x outputsize
        # outputsize = (inputsize - kernel_size + stride)/stride
        # (round up if not an integer)

        # conv1 : 80 x 80 -> 40 x 40
        self.conv1 = nn.Conv2d(2, 4, kernel_size=2, stride=2)
        # conv2 : 40 x 40 -> 20 x 20
        self.conv2 = nn.Conv2d(4, 8, kernel_size=2, stride=2)
        # conv3 : 20 x 20 -> 10 x 10
        self.conv3 = nn.Conv2d(8, 16, kernel_size=2, stride=2)
        # conv4 : 10 x 10 ->  5 x  5
        self.conv4 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.size = 32 * 5 * 5

        # 1 fully connected layer
        self.fc1 = nn.Linear(self.size, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sig(self.fc3(x))
        return x"""

    def __init__(self):
        super(Policy, self).__init__()
        # 80x80x2 to 38x38x4
        # 2 channel from the stacked frame
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False)
        # 38x38x4 to 9x9x32
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.size = 9 * 9 * 16

        # two fully connected layer
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)

        # Sigmoid to
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        return self.sig(self.fc2(x))
