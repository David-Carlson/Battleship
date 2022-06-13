from dataclasses import dataclass
from typing import Final
from typing import Tuple, Iterator
import numpy as np
from numpy import ndarray
import random as r
# https://en.wikipedia.org/wiki/Box-drawing_character


@dataclass(frozen=True)
class ShipDetail(object):
    """Used to store the hidden arrangement of ships."""
    length: int
    y: int
    x: int
    is_vert: bool


# https://datagenetics.com/blog/december32011/index.html
class Battleship(object):
    def __init__(self):
        self.size: Final = 10
        self.shots_fired = 0
        self.ships: Final = [2, 3, 3, 4, 5]
        self.ship_details: set[ShipDetail] = set()
        self.sunk_ships: set[ShipDetail] = set()

        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.solution = np.zeros((self.size, self.size), dtype=int)

        self.sol_symbols: Final = {-1: '#', 0: '·', 1: '▓', 2: '█', 3: '▓', 4: '▒', 5: '░'}
        self.grid_symbols: Final = {-1: '0', 0: '·', 1: '╳', 2: '░'}
        self.target_tiles = set()

        self.parity_boards = {p: self.get_parity_indices(p) for p in self.ships}

        self.restart()

    def restart(self, debug=False):
        """Resets the game so it can be played again. If set to debug, the ship placements aren't random."""
        self.shots_fired = 0
        self.ship_details: set[ShipDetail] = set()
        self.sunk_ships: set[ShipDetail] = set()

        self.grid[:, :] = 0
        self.solution[:, :] = 0
        if debug:
            self.mock_place_ships()
        else:
            self.place_ships()
        self.target_tiles = set()

    def place_ships(self):
        """Randomly places the ships and adds their info to ship_details."""
        for s in self.ships:
            not_placed = True
            while not_placed:
                is_vert = r.choice([True, False])
                x = r.randrange(self.size) if is_vert else r.randrange(self.size - s)
                y = r.randrange(self.size - s) if is_vert else r.randrange(self.size)
                ship_slices = self.solution[y:y + s, x] if is_vert else self.solution[y, x:x + s]
                if sum(ship_slices) == 0:
                    self.ship_details.add(ShipDetail(s, y, x, is_vert))
                    if is_vert:
                        self.solution[y:y + s, x] = s
                    else:
                        self.solution[y, x:x + s] = s
                    not_placed = False

    def mock_place_ships(self):
        """Used to set ships to a specific pattern - useful for testing."""
        self.ship_details.add(ShipDetail(2, 2, 9, True))
        self.ship_details.add(ShipDetail(3, 1, 5, True))
        self.ship_details.add(ShipDetail(3, 6, 3, False))
        self.ship_details.add(ShipDetail(4, 9, 3, False))
        self.ship_details.add(ShipDetail(5, 5, 0, False))

        for s in self.ship_details:
            y = s.y
            x = s.x
            length = s.length
            if s.is_vert:
                self.solution[y:y + length, x] = length
                # self.grid[y:y + length, x] = 1
            else:
                self.solution[y, x:x + length] = length
                # self.grid[y, x:x + length] = 1

    def play_game(self, use_strategy: bool, verbose=True, print_probs=False) -> int:
        """Play a complete game with the given settings, returns the number of shots."""
        print()
        if verbose:
            self.print_message('Playing field', '╒', '╕', ending=' ')
            self.print_message('Solution', '╒', '╕', ending=' ')
            if print_probs:
                self.print_message('Likelihood of ship', '╒', '╕')
            else:
                print()
            self.print_both(True)

        while not self.is_game_over():
            self.take_turn(use_strategy, verbose, print_probs)
        if verbose:
            print(f"Game took {self.shots_fired} turns")
        turns = self.shots_fired
        self.restart()
        return turns

    def take_turn(self, use_strategy, verbose=True, print_probs=False):
        """Decides whether to hunt or target ships based on if target_tiles exist. Can also randomly fire."""
        self.shots_fired += 1
        if use_strategy:
            if len(self.target_tiles) == 0:
                if verbose:
                    print('Hunting for ships')
                self.hunt_ships(verbose)
            else:
                if verbose:
                    print('Targeting ships')
                self.target_ships(verbose)
        else:
            self.take_random_shot(verbose)

        if verbose:
            self.print_both(print_probs=print_probs and use_strategy and not self.is_game_over())

    def hunt_ships(self, verbose):
        """Counts the likelihood of each cell having a ship and fires at the most likely.
        If hit, it searches for potential neighbors to start targeting mode."""
        parity = self.get_parity()
        p = self.get_cell_counts(is_hunting=True, parity=parity)

        (y, x) = self.choose_cell_with_probs(p)

        if verbose:
            print(f'Shooting at {y}, {x} with parity {parity}')

        self.grid[y, x] = 1 if self.solution[y, x] > 0 else -1
        if self.is_ship_sunk():
            if verbose:
                print('Sunk a ship!')
        elif self.grid[y, x] == 1:
            n = set(self.get_neighbors(y, x))
            if n:
                self.target_tiles = n
                if verbose:
                    print(f'Found a ship, considering: {n}')

    def target_ships(self, verbose):
        """Searches the most likely of the target tiles, expanding the list for every successful hit."""
        if verbose:
            print(f'Targeting: {self.target_tiles}')
        p = self.get_cell_counts(is_hunting=False, parity=self.get_parity())

        (y, x) = self.choose_cell_with_probs(p)
        self.target_tiles.remove((y, x))
        if verbose:
            print(f'Decided on {y},{x}')
        self.grid[y, x] = 1 if self.solution[y, x] > 0 else -1
        if self.is_ship_sunk():
            if verbose:
                print('Finished off a ship!')
            self.target_tiles = set(self.get_cells_near_hits(self.target_tiles))
        elif self.grid[y, x] == 1:
            if verbose:
                print('Shot a targeted ship')
            n = set(self.get_neighbors(y, x))
            if n:
                self.target_tiles.update(n)

    def apply_target_mask(self, p: ndarray) -> ndarray:
        """Given cell counts p, this keeps only the values aligning with target_tiles."""
        mask = np.zeros((self.size, self.size), dtype=float)
        for y, x in self.target_tiles:
            mask[y, x] = 1
        return np.multiply(p, mask)

    def is_game_over(self) -> bool:
        """Returns True when every cell under a ship has been hit"""
        for s in self.ship_details:
            y, x = s.y, s.x
            length = s.length
            if s.is_vert:
                if sum(self.grid[y:y + length, x]) != length:
                    return False
            else:
                if sum(self.grid[y, x:x + length]) != length:
                    return False
        return True

    def is_ship_sunk(self) -> bool:
        """Eliminates a ship when it has been sunk, which can change parity. Returns True when a ship is sunk."""
        for ship in set(self.ship_details):
            (y, x, length, is_vert) = ship.y, ship.x, ship.length, ship.is_vert
            ship_slice = self.grid[y:y + length, x] if is_vert else self.grid[y, x:x + length]
            if sum(ship_slice) == length:
                self.ship_details.remove(ship)
                self.sunk_ships.add(ship)
                if is_vert:
                    self.grid[y:y + length, x] = 2
                else:
                    self.grid[y, x:x + length] = 2
                return True
        return False

    def take_random_shot(self, verbose=True):
        """Randomly fires at the grid - a naive implementation of the game."""
        not_found = True
        while not_found:
            y = r.randrange(self.size)
            x = r.randrange(self.size)
            if self.grid[y, x] == 0:
                if verbose:
                    print(f'Shooting at {y},{x} on turn {self.shots_fired}, ignoring parity {self.get_parity()}')
                self.grid[y, x] = 1 if self.solution[y, x] > 0 else -1
                if self.is_ship_sunk():
                    if verbose:
                        print('Sunk a ship!')
                not_found = False

    def get_cell_counts(self, is_hunting, parity) -> ndarray:
        """Returns a grid where the count is how many times any remaining ship can fit on a given cell.
        If hunting, a checkerboard parity is applied, eliminating most options for efficiency.
        If targeting, only the cells that are adjacent to hits are returned.
        In either case, only the  most likely cells are kept as non-zero values."""
        p = np.zeros((self.size, self.size), dtype=float)

        for ship in self.ship_details:
            for y in range(self.size - ship.length + 1):
                for x in range(self.size):
                    ship_slice = self.grid[y:y + ship.length, x]
                    # If there's no misses or sunk ships, add 1 to the cell
                    if not (np.any(ship_slice <= -1) or np.any(ship_slice >= 2)):
                        if is_hunting:
                            p[y:y + ship.length, x] += 1
                        else:
                            bonus = ship_slice[ship_slice == 1].sum()
                            p[y:y + ship.length, x] += 1 * bonus ** 3
        for ship in self.ship_details:
            for y in range(self.size):
                for x in range(self.size - ship.length + 1):
                    ship_slice = self.grid[y, x:x + ship.length]
                    # If there's no misses or sunk ships, add 1 to the cell
                    if not (np.any(ship_slice <= -1) or np.any(ship_slice >= 2)):
                        if is_hunting:
                            p[y, x:x + ship.length] += 1
                        else:
                            bonus = ship_slice[ship_slice == 1].sum()
                            p[y, x:x + ship.length] += 1 * bonus ** 3

        if is_hunting:
            p = np.multiply(p, self.get_parity_indices(parity))
        else:
            p = self.apply_target_mask(p)

        p[p < (p.max(initial=0) - 1)] = 0
        p[self.grid == 1] = 0
        p[self.grid == -1] = 0

        if p.sum() <= 0:
            print('negative sum')
            print(p)
            p = np.zeros((self.size, self.size), dtype=int)
            p[self.grid == 0] = 1
        return p

    def choose_cell_with_probs(self, p: ndarray) -> Tuple[int, int]:
        """Given cell counts p, this randomly returns the chosen (y,x) cell."""
        indices = np.arange(100, dtype=int)
        idx = np.random.choice(indices, p=p.flatten() / p.sum())
        return idx // self.size, idx % self.size

    def get_parity_indices(self, parity: int) -> ndarray:
        """Returns a checkerboard mask of 1 and 0 indicating where to choose from, avoiding choosing all cells."""
        indices = np.zeros((self.size, self.size), dtype=int)
        for r in range(parity):
            indices[r::parity, r::parity] = 1
        return indices

    def get_parity(self) -> int:
        """Returns the parity of the board (length of smallest ship).
        If the smallest ship if 5, only 1 in 5 adjacent cells need searching."""
        return min([s.length for s in self.ship_details])

    def probs_as_printable(self) -> ndarray:
        """Forces the cell-count array into the range 0-9 to allow easy printing."""
        p = self.get_cell_counts(is_hunting=len(self.target_tiles) == 0, parity=self.get_parity())
        p = np.abs(p / p.max(initial=1)) * 10
        p = np.floor(p)
        p[p > 9] = 9
        return p.astype(int)

    def get_neighbors(self, y, x) -> Iterator[Tuple[int, int]]:
        """For a given cell y,x, this returns an Iterator of all unchecked neighbor cells."""
        neighbors = ((-1, 0), (0, 1), (1, 0), (0, -1))
        for y_diff, x_diff in neighbors:
            y_point, x_point = y + y_diff, x + x_diff
            if y_point in range(0, self.size) and x_point in range(0, self.size):
                if self.grid[y_point, x_point] == 0:
                    yield y_point, x_point
        return

    def get_cells_near_hits(self, tiles) -> Iterator[Tuple[int, int]]:
        """For all remaining target_tiles, this returns only those next to an unaccounted for hit.
        This is used in target mode after a ship has been sunk to ignore target_tiles next to sunk ships."""
        neighbors = ((-1, 0), (0, 1), (1, 0), (0, -1))
        for (y, x) in tiles:
            for y_diff, x_diff in neighbors:
                y_point, x_point = y + y_diff, x + x_diff
                if y_point in range(0, self.size) and x_point in range(0, self.size):
                    if self.grid[y_point, x_point] == 1:
                        yield y, x
        return

    def print_solution(self):
        """Prints the solution board of ships"""
        self._print(self.solution, self.sol_symbols)

    def print_grid(self):
        """Prints the grid showing all hits/misses/sunk ships"""
        self._print(self.grid, self.grid_symbols)

    def print_both(self, print_probs=False):
        """Prints the hit grid and solution grid, with optional probability grid"""
        fill = ' '
        probs = self.probs_as_printable() if print_probs else []
        self.print_border('═', '╔', '╗' + fill, ending='')
        if print_probs:
            self.print_border('═', '╔', '╗' + fill, ending='')
        self.print_border('═', '╔', '╗')
        for y in range(self.size):
            print('║ ', end='')
            for x in range(self.size):
                sym = self.grid_symbols[self.grid[y, x]]
                print(f'{sym} ', end='')
            print('║' + fill + '║ ', end='')
            for x in range(self.size):
                sym = self.sol_symbols[self.solution[y, x]]
                print(f'{sym} ', end='')
            if print_probs:
                print('║' + fill + '║ ', end='')
                for x in range(self.size):
                    sym = probs[y, x]
                    sym = sym if sym > 0 else '·'
                    print(f'{sym} ', end='')

            print('║')
        self.print_border('═', '╚', '╝' + fill, ending='')
        if print_probs:
            self.print_border('═', '╚', '╝' + fill, ending='')
        self.print_border('═', '╚', '╝')

    def _print(self, src, symbols):
        """Prints a given src array and accompanying symbol dictionary."""
        self.print_border('═', '╔', '╗')
        for y in range(self.size):
            print('║ ', end='')
            for x in range(self.size):
                sym = symbols[src[y, x]]
                print(f"{sym} ", end='')
            print('║')
        self.print_border('═', '╚', '╝')

    def print_border(self, sym, start='', stop='', ending='\n'):
        """Prints the start symbol and end symbol with enough sym symbols between to match the grid size."""
        print(f'{start}{sym * (2 * self.size + 1)}{stop}', end=ending)

    def print_message(self, msg, start='', stop='', ending='\n'):
        print(f'{start}{msg.center(2 * self.size + 1)}{stop}', end=ending)



if __name__ == '__main__':
    bs = Battleship()
    bs.play_game(use_strategy=True, verbose=True, print_probs=True)

    # TODO

    # Optimize:
    # Profile?
    # Save parity boards
    # save zero boards?
    # Remove reassignments?
    # Use threading
