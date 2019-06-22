import random
import numpy as np
from copy import deepcopy

class Tictactoe:

    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]

    def reset(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        return self.board

    # Returns state, reward, is over
    def make_move(self, player, x, y):
        self.board[x][y] = player
        if self.is_winner(player, x, y):
            if player == 'X':
                return self.board, 1, True
            return self.board, -1, True

        return self.board, 0, not any([' ' in r for r in self.board])

    def is_winner(self, player, x, y):
        p = [player] * 3
        if p == self.board[x]:
            return True
        if p == [r[y] for r in self.board]:
            return True

        diag = [r[i] for i, r in enumerate(self.board)]
        if p == diag:
            return True

        diag = [r[2-i] for i, r in enumerate(self.board)]
        return p == diag

    def print_board(self):
        print('==========================\n')
        first = True
        for r in self.board:
            if not first:
                print('--------------')
            first = False
            print(r[0], '|', r[1], '|', r[2])
        print('==========================\n')

    # 0 1 2
    # 3 4 5
    # 6 7 8

    def encode_state(self, state):
        value = 0
        for x in range(3):
            for y in range(3):
                id = x * 3 + y
                three_pow = 3 ** id
                coefficent = 0
                if state[x][y] == 'X':
                    coefficent = 1
                if state[x][y] == 'O':
                    coefficent = 2
                value += three_pow * coefficent
        return value

    def random_move(self,state):
        moves = []
        for x in range(3):
            for y in range(3):
                if state[x][y] == ' ':
                    moves.append(x * 3 + y)

        return moves[random.randint(0, len(moves) - 1)]

    def ai_move(self):
        for x in range(3):
            for y in range(3):
                if self.board[x][y] == ' ':
                    self.board[x][y] = 'X'
                    if self.is_winner('X', x, y):
                        self.board[x][y] = ' '
                        return x * 3 + y
                    self.board[x][y] = ' '

        return self.random_move(self.board)

def main():
    q_table = [[0] * 9 for _ in range(3 ** 10)]
    gamma = .99
    epsilon = .2
    epochs = 100000
    lr = .1

    for ep in range(epochs):
        if ep % 10000 == 0:
            print(ep)

        env = Tictactoe()
        done = False
        state = env.reset()
        states = []
        moves = []
        rewards = []

        while not done:
            state_id = env.encode_state(state)
            q_row = q_table[state_id]

            for x in range(3):
                for y in range(3):
                    if state[x][y] != ' ':
                        q_row[x * 3 + y] = -100

            move = np.argmax(q_row)
            if random.random() < epsilon:
                move = env.random_move(state)

            moves.append(move)
            states.append(deepcopy(state))
            state, reward, done = env.make_move('X', move //3, move % 3)
            rewards.append(reward)

            if done:
                break

            move = env.ai_move()
            state, reward, done = env.make_move('O', move//3, move % 3)
            rewards[-1] += reward

        rewards = rewards[::-1]
        moves = moves[::-1]
        states = states[::-1]
        for i in range(len(states)):
            state_id = env.encode_state(states[i])
            if i == 0:
                q_table[state_id][moves[i]] = q_table[state_id][moves[i]] + lr * (rewards[i] - q_table[state_id][moves[i]])
            else:
                q_table[state_id][moves[i]] = q_table[state_id][moves[i]] + lr * (gamma * max(q_table[env.encode_state(states[i - 1])]) - q_table[state_id][moves[i]])


    state = env.reset()
    done = False
    while not done:
        env.print_board()
        state_id = env.encode_state(state)
        move = np.argmax(q_table[state_id])
        print(q_table[state_id])
        state, reward, done = env.make_move('X', move // 3, move % 3)
        env.print_board()
        if done:
            break
        move = int(input('Select Move\n'))
        state, reward, done = env.make_move('O', move // 3, move % 3)
    env.print_board()

main()
