from multiprocessing import Process
from threading import Thread
import os
import math
import time
from Battleship import Battleship


def calculate(n: int, rounds = 1):
    bs = Battleship()
    times = []
    avgs = []
    for _ in range(rounds):
        start = time.perf_counter()
        avg = sum(bs.play_game(use_strategy=True, verbose=False, print_probs=True) for _ in range(n)) / n
        avgs.append(avg)
        print(f'Average shots: {avg}')
        end = time.perf_counter()
        times.append(end - start)
        # print(f'{n} Strategic games took {end - start}')

    print(f'\nTotal Time average: {sum(times) / len(times)}')


if __name__ == "__main__":
    total_start = time.perf_counter()
    n = 500
    print(f'There are {os.cpu_count()} cores each calculating {n} games')

    processes = [Process(target=calculate, args=(n, 1)) for _ in range(os.cpu_count())]
    for p in processes:
        p.start()

    for p in processes:
        p.join()
    print('Done with thread test')
    total_end = time.perf_counter()
    print(f'Time to complete all threads: {total_end - total_start}')
