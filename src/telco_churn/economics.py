"""The commercial assumptions that turn probabilities into decisions.

These are *not* findings. They are placeholders that must be replaced with the
business's own numbers before any value figure is quoted externally. Keeping
them in one object makes the sensitivity analysis in notebook 02 possible and
makes it obvious what was assumed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Economics:
    value_saved: float = 300.0   # margin retained when a would-be churner is saved
    contact_cost: float = 10.0   # cost of one retention contact
    save_rate: float = 0.30      # share of contacted would-be churners the offer saves

    @property
    def breakeven_probability(self) -> float:
        """Contact is worth it above this churn probability."""
        return self.contact_cost / (self.save_rate * self.value_saved)

    def relative_value(self, monthly_charges):
        """Value at risk per customer, proportional to the bill.

        Scaled so the *average* customer is worth ``value_saved``. This keeps
        the targeting layer on the same money scale as the threshold analysis
        while letting a high-bill customer be worth more than a low-bill one.
        """
        import numpy as np

        charges = np.asarray(monthly_charges, dtype=float)
        mean = charges.mean()
        if not np.isfinite(mean) or mean <= 0:
            raise ValueError("monthly charges must have a positive mean")
        return charges / mean * self.value_saved

    def as_dict(self) -> dict:
        return asdict(self)
