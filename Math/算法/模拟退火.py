import math
import random
from typing import Callable, List, Sequence, Tuple


def simulated_annealing(
	objective: Callable[[Sequence[float]], float],
	initial_state: Sequence[float],
	bounds: Sequence[Tuple[float, float]],
	initial_temperature: float = 100.0,
	cooling_rate: float = 0.95,
	min_temperature: float = 1e-3,
	iterations_per_temperature: int = 100,
) -> Tuple[List[float], float]:
	"""Minimize an objective function with simulated annealing."""

	if len(initial_state) != len(bounds):
		raise ValueError("initial_state and bounds must have the same length")

	def clip_state(state: Sequence[float]) -> List[float]:
		clipped: List[float] = []
		for value, (lower, upper) in zip(state, bounds):
			clipped.append(max(lower, min(upper, value)))
		return clipped

	def neighbor(state: Sequence[float], temperature: float) -> List[float]:
		scale = max(temperature / initial_temperature, 1e-6)
		candidate = []
		for value, (lower, upper) in zip(state, bounds):
			step = random.uniform(-1.0, 1.0) * (upper - lower) * 0.1 * scale
			candidate.append(value + step)
		return clip_state(candidate)

	current_state = clip_state(initial_state)
	current_value = objective(current_state)
	best_state = current_state[:]
	best_value = current_value

	temperature = initial_temperature
	while temperature > min_temperature:
		for _ in range(iterations_per_temperature):
			candidate_state = neighbor(current_state, temperature)
			candidate_value = objective(candidate_state)
			delta = candidate_value - current_value

			if delta < 0 or random.random() < math.exp(-delta / temperature):
				current_state = candidate_state
				current_value = candidate_value

				if current_value < best_value:
					best_state = current_state[:]
					best_value = current_value

		temperature *= cooling_rate

	return best_state, best_value


def rastrigin(x: Sequence[float]) -> float:
	n = len(x)
	return 10 * n + sum(val * val - 10 * math.cos(2 * math.pi * val) for val in x)


if __name__ == "__main__":
	random.seed(42)

	init = [5.0, -3.0]
	search_bounds = [(-5.12, 5.12), (-5.12, 5.12)]

	best_x, best_y = simulated_annealing(
		objective=rastrigin,
		initial_state=init,
		bounds=search_bounds,
		initial_temperature=100.0,
		cooling_rate=0.95,
		min_temperature=1e-3,
		iterations_per_temperature=200,
	)

	print("best_x =", best_x)
	print("best_value =", best_y)
