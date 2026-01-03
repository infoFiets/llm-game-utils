"""Budget tracking to prevent excessive API spending.

This module provides the BudgetTracker class to monitor and limit API costs
during development and testing. It helps prevent accidentally spending too
much money from bugs, infinite loops, or runaway processes.
"""

import json
import logging
import os
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional

from ..exceptions import BudgetExceededError

logger = logging.getLogger(__name__)


class BudgetTracker:
    """Track API spending against daily and session budgets.

    Prevents runaway costs by enforcing spending limits. Tracks both
    daily totals (persisted across sessions) and session totals (reset
    when the tracker is created).

    Example:
        ```python
        # Create a budget tracker
        tracker = BudgetTracker(
            daily_budget=10.0,      # $10 per day
            session_budget=2.0,     # $2 per session
            warning_threshold=0.8   # Warn at 80%
        )

        # Check before making an expensive call
        estimated_cost = 0.05
        if tracker.check_budget(estimated_cost):
            response = client.query(...)
            tracker.add_cost(response.cost)
        else:
            print("Budget exceeded!")

        # Get status
        print(tracker.get_status_report())
        ```
    """

    def __init__(
        self,
        daily_budget: float = 10.0,
        session_budget: Optional[float] = None,
        warning_threshold: float = 0.8,
        storage_dir: Optional[str] = None,
        enforce_limits: bool = True
    ):
        """Initialize the budget tracker.

        Args:
            daily_budget: Maximum to spend per day in dollars (default: $10)
            session_budget: Maximum to spend this session in dollars (optional)
            warning_threshold: Warn when this fraction of budget is used (0.0-1.0)
            storage_dir: Directory to store budget data (default: ~/.llm_game_utils)
            enforce_limits: If True, raise exception when budget exceeded.
                          If False, just log warnings.

        Raises:
            ValueError: If budgets or threshold are invalid
        """
        if daily_budget <= 0:
            raise ValueError("daily_budget must be positive")

        if session_budget is not None and session_budget <= 0:
            raise ValueError("session_budget must be positive")

        if not 0.0 <= warning_threshold <= 1.0:
            raise ValueError("warning_threshold must be between 0.0 and 1.0")

        self.daily_budget = daily_budget
        self.session_budget = session_budget
        self.warning_threshold = warning_threshold
        self.enforce_limits = enforce_limits

        # Session spending (resets when tracker is recreated)
        self.session_spent = 0.0

        # Storage setup
        if storage_dir is None:
            storage_dir = os.path.expanduser("~/.llm_game_utils")

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.budget_file = self.storage_dir / "budget_tracker.json"

        # Load or initialize daily spending
        self._load_daily_data()

        logger.info(
            f"BudgetTracker initialized: "
            f"daily=${self.daily_budget}, session=${self.session_budget}, "
            f"enforce={self.enforce_limits}"
        )

    def _load_daily_data(self) -> None:
        """Load daily spending data from disk."""
        today = date.today().isoformat()

        if self.budget_file.exists():
            try:
                with open(self.budget_file, 'r') as f:
                    data = json.load(f)

                # Check if the stored date is today
                if data.get('date') == today:
                    self.daily_spent = data.get('spent', 0.0)
                    logger.debug(f"Loaded daily spending: ${self.daily_spent:.4f}")
                else:
                    # It's a new day, reset
                    self.daily_spent = 0.0
                    logger.info("New day - daily budget reset")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load budget data: {e}. Starting fresh.")
                self.daily_spent = 0.0
        else:
            self.daily_spent = 0.0
            logger.debug("No existing budget data found")

    def _save_daily_data(self) -> None:
        """Save daily spending data to disk."""
        today = date.today().isoformat()

        data = {
            'date': today,
            'spent': self.daily_spent,
            'last_updated': datetime.now().isoformat()
        }

        try:
            with open(self.budget_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save budget data: {e}")

    def check_budget(self, estimated_cost: float) -> bool:
        """Check if we can afford this API call.

        Args:
            estimated_cost: Estimated cost of the upcoming API call in dollars

        Returns:
            True if within budget, False if would exceed budget

        Raises:
            BudgetExceededError: If enforce_limits=True and budget would be exceeded
        """
        # Check daily budget
        if self.daily_spent + estimated_cost > self.daily_budget:
            message = (
                f"Daily budget would be exceeded: "
                f"${self.daily_spent + estimated_cost:.4f} > ${self.daily_budget:.4f}"
            )

            if self.enforce_limits:
                raise BudgetExceededError('daily', self.daily_budget, self.daily_spent + estimated_cost)
            else:
                logger.warning(message)
                return False

        # Check session budget if set
        if self.session_budget is not None:
            if self.session_spent + estimated_cost > self.session_budget:
                message = (
                    f"Session budget would be exceeded: "
                    f"${self.session_spent + estimated_cost:.4f} > ${self.session_budget:.4f}"
                )

                if self.enforce_limits:
                    raise BudgetExceededError('session', self.session_budget, self.session_spent + estimated_cost)
                else:
                    logger.warning(message)
                    return False

        # Check if we're approaching the warning threshold
        if self.is_near_limit():
            daily_pct = (self.daily_spent / self.daily_budget) * 100
            logger.warning(
                f"Approaching budget limit: {daily_pct:.1f}% of daily budget used"
            )

        return True

    def add_cost(self, actual_cost: float) -> None:
        """Record the actual cost of an API call.

        Args:
            actual_cost: Actual cost in dollars
        """
        if actual_cost < 0:
            logger.warning(f"Negative cost ignored: ${actual_cost}")
            return

        self.daily_spent += actual_cost
        self.session_spent += actual_cost

        logger.debug(
            f"Cost recorded: ${actual_cost:.4f} | "
            f"Daily: ${self.daily_spent:.4f}/{self.daily_budget:.4f} | "
            f"Session: ${self.session_spent:.4f}"
            + (f"/{self.session_budget:.4f}" if self.session_budget else "")
        )

        # Save updated daily total
        self._save_daily_data()

    def get_remaining_budget(self) -> Dict[str, float]:
        """Get remaining budget amounts.

        Returns:
            Dictionary with daily and session remaining budgets and spending:
            {
                "daily_remaining": float,
                "session_remaining": float or None,
                "daily_spent": float,
                "session_spent": float
            }
        """
        return {
            "daily_remaining": max(0, self.daily_budget - self.daily_spent),
            "session_remaining": (
                max(0, self.session_budget - self.session_spent)
                if self.session_budget is not None
                else None
            ),
            "daily_spent": self.daily_spent,
            "session_spent": self.session_spent
        }

    def reset_session(self) -> None:
        """Reset session spending (keeps daily total)."""
        old_session = self.session_spent
        self.session_spent = 0.0
        logger.info(f"Session budget reset (was ${old_session:.4f})")

    def is_near_limit(self) -> bool:
        """Check if spending is near the warning threshold.

        Returns:
            True if either daily or session spending is >= warning threshold
        """
        daily_fraction = self.daily_spent / self.daily_budget

        if daily_fraction >= self.warning_threshold:
            return True

        if self.session_budget is not None:
            session_fraction = self.session_spent / self.session_budget
            if session_fraction >= self.warning_threshold:
                return True

        return False

    def get_status_report(self) -> str:
        """Generate a human-readable budget status report.

        Returns:
            Formatted string with budget information
        """
        daily_pct = (self.daily_spent / self.daily_budget) * 100
        daily_remaining = self.daily_budget - self.daily_spent

        lines = [
            "=== Budget Status ===",
            f"Daily Budget:   ${self.daily_budget:.2f}",
            f"Daily Spent:    ${self.daily_spent:.4f} ({daily_pct:.1f}%)",
            f"Daily Remaining: ${daily_remaining:.4f}",
        ]

        if self.session_budget is not None:
            session_pct = (self.session_spent / self.session_budget) * 100
            session_remaining = self.session_budget - self.session_spent
            lines.extend([
                f"",
                f"Session Budget:   ${self.session_budget:.2f}",
                f"Session Spent:    ${self.session_spent:.4f} ({session_pct:.1f}%)",
                f"Session Remaining: ${session_remaining:.4f}",
            ])
        else:
            lines.append(f"Session Spent:    ${self.session_spent:.4f}")

        # Add warning if near limit
        if self.is_near_limit():
            lines.append("")
            lines.append("⚠️  WARNING: Approaching budget limit!")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation of the budget tracker."""
        return self.get_status_report()

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"BudgetTracker(daily=${self.daily_budget:.2f}, "
            f"session=${self.session_budget if self.session_budget else 'None'}, "
            f"daily_spent=${self.daily_spent:.4f}, "
            f"session_spent=${self.session_spent:.4f})"
        )
