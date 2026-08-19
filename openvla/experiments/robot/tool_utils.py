"""Utility helpers shared across robot experiment tooling."""

from __future__ import annotations

import math

from typing import Sequence, List, Optional


def compute_dynamic_threshold(
	step_idx: int,
	total_steps: int,
	start: float,
	lower: float,
	schedule: str = "linear",
) -> float:


	step_idx = max(step_idx, 0)
	total_steps = max(total_steps, 1)

	if start < lower:
		raise ValueError("start threshold must be >= lower threshold")

	if schedule == "linear":
		epsilon = 1e-30
		shape_p = 3.0
		target_step = float(total_steps)
		tau = target_step / ((-math.log(epsilon)) ** (1.0 / shape_p))
		step = max(float(step_idx), 0.0)
		decay = math.exp(-((step / tau) ** shape_p))
		value = lower + (start - lower) * decay
	elif schedule == "exponential":
		if start == 0:
			return lower
		ratio = lower / start if start else 0.0
		ratio = max(min(ratio, 1.0), 0.0)
		decay = ratio ** (step_idx / total_steps)
		value = start * decay
	else:
		raise ValueError(f"Unsupported threshold schedule: {schedule}")

	return max(lower, value)


def kalman_predict_from_history(
	history: Sequence[float],
	*,
	initial_estimate_error: float = 10.0,
	process_variance: float = 1.0,
	measurement_variance: float = 1e-3,
) -> float:
	"""Run a 1D Kalman filter over a history window and predict the next value."""

	if not history:
		raise ValueError("history must contain at least one element")

	state_estimate = float(history[0])
	estimate_error = float(initial_estimate_error)

	for measurement in history:
		predicted_state = state_estimate
		predicted_error = estimate_error + process_variance

		innovation = float(measurement) - predicted_state
		innovation_den = predicted_error + measurement_variance
		if innovation_den <= 0:
			raise ValueError("Invalid measurement variance; denominator <= 0")

		kalman_gain = predicted_error / innovation_den
		state_estimate = predicted_state + kalman_gain * innovation
		estimate_error = (1.0 - kalman_gain) * predicted_error

	return state_estimate


def kalman_predict_7d_from_history(
	history: Sequence[Sequence[float]],
	*,
	observation: Optional[Sequence[float]] = None,
	initial_estimate_error: float = 10.0,
	process_variance: float = 1.0,
	measurement_variance: float = 1e-3,
) -> List[float]:


	if (history is None or len(history) == 0) and observation is None:
		raise ValueError("history or observation must be provided")

	# Determine dimensionality
	if observation is not None and len(observation) > 0:
		dim = len(observation)
	elif history and len(history) > 0 and len(history[0]) > 0:
		dim = len(history[0])
	else:
		# Default to 7 if nothing else known
		dim = 7

	# Initialize state and error
	if history and len(history) > 0:
		init_vec = list(history[0])
		if len(init_vec) != dim:
			raise ValueError("history vectors must all have the same dimensionality")
		state_estimate: List[float] = [float(v) for v in init_vec]
	else:
		# If no history, bootstrap from the observation (or zeros)
		if observation is not None and len(observation) == dim:
			state_estimate = [float(v) for v in observation]
		else:
			state_estimate = [0.0] * dim

	estimate_error: List[float] = [float(initial_estimate_error)] * dim
	Q = float(process_variance)
	R = float(measurement_variance)

	def _update(meas: Sequence[float]):
		# One full predict+update for a vector measurement
		for i in range(dim):
			predicted_state = state_estimate[i]
			predicted_error = estimate_error[i] + Q
			innovation = float(meas[i]) - predicted_state
			den = predicted_error + R
			# Guard against degenerate values
			if den <= 0:
				# Fallback: skip update for this dim
				state_estimate[i] = predicted_state
				estimate_error[i] = max(predicted_error, 1e-9)
				continue
			K = predicted_error / den
			state_estimate[i] = predicted_state + K * innovation
			estimate_error[i] = (1.0 - K) * predicted_error

	# Consume historical vectors
	if history:
		for vec in history:
			if len(vec) != dim:
				raise ValueError("All history vectors must share the same dimensionality")
			_update(vec)

	# Apply current-step observation if given
	if observation is not None:
		if len(observation) != dim:
			raise ValueError("observation has mismatched dimensionality")
		_update(observation)

	return state_estimate



# main
if __name__ == "__main__":
	pass