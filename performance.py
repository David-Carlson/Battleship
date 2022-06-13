from Battleship import Battleship
import time

if __name__ == '__main__':
    print('Starting performance check')
    bs = Battleship()

    n = 8 * 500

    times = []
    avgs = []
    rounds = 1
    total_start = time.perf_counter()
    for _ in range(rounds):
        start = time.perf_counter()
        avg = sum(bs.play_game(use_strategy=True, verbose=False, print_probs=True) for _ in range(n)) / n
        avgs.append(avg)
        print(f'Average shots: {avg}')
        end = time.perf_counter()
        times.append(end-start)
        print(f'{n} Strategic games took {end - start}')

    print(f'\nTotal Time average: {sum(times)/len(times)}')
    print(f'Total Average shots: {sum(avgs)/len(avgs)}\n')


    # print('Starting random game/s')
    # start = time.perf_counter()
    # avg = sum(bs.play_game(use_strategy=False, verbose=False, print_probs=True) for _ in range(n)) / n
    # print(f'Average shots: {avg}')
    # end = time.perf_counter()
    # print(f'{n} Random games took {end - start}')

    print('\nPerformance check complete!')
    print(f'Time to complete performance check {time.perf_counter() - total_start}')

