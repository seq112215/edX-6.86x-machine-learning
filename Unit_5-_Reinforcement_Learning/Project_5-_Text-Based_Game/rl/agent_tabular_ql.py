"""Tabular QL agent"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import framework
import utils

DEBUG = False

GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 200
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.1  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


def epsilon_greedy(state_1, state_2, q_func, epsilon):
    """Returns an action selected by an epsilon-Greedy exploration policy

    Args:
        state_1, state_2 (int, int): two indices describing the current state
        q_func (np.ndarray): current Q-function
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    """ My solution:
    if np.random.random() < epsilon:
        action_index, object_index = np.random.randint(0, NUM_ACTIONS), \
                                     np.random.randint(0, NUM_OBJECTS)
    else:
        action_index, object_index = np.unravel_index(np.argmax(q_func[state_1, state_2]),
                                                      (NUM_ACTIONS, NUM_OBJECTS))

    return action_index, object_index
    """
    # Instructor's solution: (same)
    coin = np.random.random_sample()

    if coin < epsilon:
        action_index = np.random.randint(NUM_ACTIONS)
        object_index = np.random.randint(NUM_OBJECTS)
    else:
        q_values = q_func[state_1, state_2, :, :]
        action_index, object_index = np.unravel_index(np.argmax(
            q_values, axis=None), q_values.shape)

    return action_index, object_index


def tabular_q_learning(q_func, current_state_1, current_state_2, action_index,
                       object_index, reward, next_state_1, next_state_2,
                       terminal):
    """Update q_func for a given transition

    Args:
        q_func (np.ndarray): current Q-function
        current_state_1, current_state_2 (int, int): two indices describing the current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_1, next_state_2 (int, int): two indices describing the next state
        terminal (bool): True if this epsiode is over

    Returns:
        None
    """
    """ My solution:
    if not terminal:
        q_func[current_state_1, current_state_2, action_index, object_index] = \
            (1 - ALPHA) * q_func[current_state_1, current_state_2,
                                 action_index, object_index] + \
            ALPHA * (reward + GAMMA * q_func[next_state_1, next_state_2].max())
    else:
        q_func[current_state_1, current_state_2, action_index, object_index] = \
            (1 - ALPHA) * q_func[current_state_1, current_state_2,
                                 action_index, object_index] + \
            ALPHA * reward

    return None  # This function shouldn't return anything
    """
    # Instructor's solution:
    if terminal:
        maxq_next_state = 0
    else:
        q_values_next_state = q_func[next_state_1, next_state_2, :, :]
        maxq_next_state = np.max(q_values_next_state)

    q_value = q_func[
        current_state_1, current_state_2, action_index, object_index]
    q_func[current_state_1, current_state_2, action_index, object_index] = \
        (1 - ALPHA) * q_value + ALPHA * (reward + GAMMA * maxq_next_state)


def run_episode(for_training):
    """ Runs one episode
    If for training, update Q function
    If for testing, computes and return cumulative discounted reward

    Args:
        for_training (bool): True if for training

    Returns:
        None
    """
    """ My solution:
    epsilon = TRAINING_EP if for_training else TESTING_EP

    epi_reward = 0

    # initialize for each episode
    current_room_desc, current_quest_desc, terminal = framework.newGame()  # string descriptions

    t = 0
    while not terminal:

        # Choose next action and execute
        room_index = dict_room_desc[current_room_desc]
        quest_index = dict_quest_desc[current_quest_desc]

        action_index, object_index = epsilon_greedy(room_index, quest_index,
                                                    q_func, epsilon)

        next_room_desc, next_quest_desc, reward, terminal = \
            framework.step_game(current_room_desc, current_quest_desc,
                                action_index, object_index)

        next_room_index = dict_room_desc[next_room_desc]
        next_quest_index = dict_quest_desc[next_quest_desc]

        if for_training:
            # update Q-function.
            tabular_q_learning(q_func, room_index, quest_index,
                               action_index, object_index,
                               reward, next_room_index, next_quest_index,
                               terminal)

        if not for_training:
            # update reward
            epi_reward += GAMMA ** t * reward
            t += 1

        # prepare next step
        current_room_desc, current_quest_desc = next_room_desc, next_quest_desc

    if not for_training:
        return epi_reward
    """
    # Instructor's solution:
    #   Difference reward section:
    #       Uses epi_reward += gamma_step * reward, then gamma_step += GAMMA
    epsilon = TRAINING_EP if for_training else TESTING_EP
    gamma_step = 1
    epi_reward = 0

    (current_room_desc, current_quest_desc, terminal) = framework.newGame()
    while not terminal:
        # Choose next action and execute
        cur_room_desc_id = dict_room_desc[current_room_desc]
        cur_quest_desc_id = dict_quest_desc[current_quest_desc]
        (action_index, object_index) = epsilon_greedy(cur_room_desc_id,
                                                      cur_quest_desc_id,
                                                      q_func, epsilon)
        (next_room_desc, next_quest_desc, reward,
         terminal) = framework.step_game(current_room_desc, current_quest_desc,
                                         action_index, object_index)

        if for_training:
            # update Q-function.
            next_room_desc_id = dict_room_desc[next_room_desc]
            next_quest_desc_id = dict_quest_desc[next_quest_desc]
            tabular_q_learning(q_func, cur_room_desc_id, cur_quest_desc_id,
                               action_index, object_index, reward,
                               next_room_desc_id, next_quest_desc_id, terminal)

        if not for_training:
            # update reward
            epi_reward = epi_reward + gamma_step * reward
            gamma_step = gamma_step * GAMMA

        # prepare next step
        current_room_desc = next_room_desc
        current_quest_desc = next_quest_desc

    if not for_training:
        return epi_reward


def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Returns array of test reward per epoch for one run"""
    global q_func
    q_func = np.zeros((NUM_ROOM_DESC, NUM_QUESTS, NUM_ACTIONS, NUM_OBJECTS))

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test


if __name__ == '__main__':
    # Data loading and build the dictionaries that use unique index for each state
    (dict_room_desc, dict_quest_desc) = framework.make_all_states_index()
    NUM_ROOM_DESC = len(dict_room_desc)
    NUM_QUESTS = len(dict_quest_desc)

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test,
                         axis=0))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Tablular: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()

    # print average reward after convergence at about 15 epochs
    print(
        np.mean(np.mean(epoch_rewards_test, axis=0)[15:]))  # 0.5143261323957701
    # Instructor's solution: 0.5135384423830186
