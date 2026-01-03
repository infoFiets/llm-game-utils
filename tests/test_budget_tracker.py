"""Tests for budget tracking functionality."""

import json
import os
import tempfile
from datetime import date
from pathlib import Path

import pytest

from llm_game_utils.tracking.budget import BudgetTracker
from llm_game_utils.exceptions import BudgetExceededError


class TestBudgetTrackerInitialization:
    """Test budget tracker initialization."""

    def test_default_initialization(self, tmp_path):
        """Test initialization with default values."""
        tracker = BudgetTracker(storage_dir=str(tmp_path))
        assert tracker.daily_budget == 10.0
        assert tracker.session_budget is None
        assert tracker.warning_threshold == 0.8
        assert tracker.enforce_limits is True
        assert tracker.daily_spent == 0.0
        assert tracker.session_spent == 0.0

    def test_custom_budgets(self, tmp_path):
        """Test initialization with custom budgets."""
        tracker = BudgetTracker(
            daily_budget=20.0,
            session_budget=5.0,
            storage_dir=str(tmp_path)
        )
        assert tracker.daily_budget == 20.0
        assert tracker.session_budget == 5.0

    def test_custom_warning_threshold(self, tmp_path):
        """Test initialization with custom warning threshold."""
        tracker = BudgetTracker(
            warning_threshold=0.5,
            storage_dir=str(tmp_path)
        )
        assert tracker.warning_threshold == 0.5

    def test_enforce_limits_false(self, tmp_path):
        """Test initialization with enforce_limits=False."""
        tracker = BudgetTracker(
            enforce_limits=False,
            storage_dir=str(tmp_path)
        )
        assert tracker.enforce_limits is False

    def test_invalid_daily_budget(self, tmp_path):
        """Test that negative daily budget raises error."""
        with pytest.raises(ValueError):
            BudgetTracker(daily_budget=-1.0, storage_dir=str(tmp_path))

        with pytest.raises(ValueError):
            BudgetTracker(daily_budget=0.0, storage_dir=str(tmp_path))

    def test_invalid_session_budget(self, tmp_path):
        """Test that negative session budget raises error."""
        with pytest.raises(ValueError):
            BudgetTracker(
                session_budget=-1.0,
                storage_dir=str(tmp_path)
            )

    def test_invalid_warning_threshold(self, tmp_path):
        """Test that invalid warning threshold raises error."""
        with pytest.raises(ValueError):
            BudgetTracker(
                warning_threshold=-0.1,
                storage_dir=str(tmp_path)
            )

        with pytest.raises(ValueError):
            BudgetTracker(
                warning_threshold=1.5,
                storage_dir=str(tmp_path)
            )


class TestBudgetTracking:
    """Test budget tracking functionality."""

    def test_add_cost(self, tmp_path):
        """Test adding costs."""
        tracker = BudgetTracker(storage_dir=str(tmp_path))
        tracker.add_cost(1.5)
        assert tracker.daily_spent == 1.5
        assert tracker.session_spent == 1.5

        tracker.add_cost(2.0)
        assert tracker.daily_spent == 3.5
        assert tracker.session_spent == 3.5

    def test_add_multiple_costs(self, tmp_path):
        """Test adding multiple costs."""
        tracker = BudgetTracker(storage_dir=str(tmp_path))
        costs = [0.1, 0.2, 0.3, 0.4, 0.5]
        for cost in costs:
            tracker.add_cost(cost)

        assert tracker.daily_spent == sum(costs)
        assert tracker.session_spent == sum(costs)

    def test_add_negative_cost(self, tmp_path):
        """Test that negative costs are ignored."""
        tracker = BudgetTracker(storage_dir=str(tmp_path))
        tracker.add_cost(1.0)
        tracker.add_cost(-0.5)  # Should be ignored

        assert tracker.daily_spent == 1.0
        assert tracker.session_spent == 1.0

    def test_check_budget_within_limit(self, tmp_path):
        """Test check_budget when within limit."""
        tracker = BudgetTracker(daily_budget=10.0, storage_dir=str(tmp_path))
        assert tracker.check_budget(5.0) is True

    def test_check_budget_at_limit(self, tmp_path):
        """Test check_budget at exact limit."""
        tracker = BudgetTracker(daily_budget=10.0, storage_dir=str(tmp_path))
        tracker.add_cost(10.0)
        assert tracker.check_budget(0.0) is True

    def test_check_budget_exceeds_daily_with_enforce(self, tmp_path):
        """Test budget check raises error when exceeded with enforce=True."""
        tracker = BudgetTracker(
            daily_budget=10.0,
            enforce_limits=True,
            storage_dir=str(tmp_path)
        )
        tracker.add_cost(9.0)

        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check_budget(2.0)

        assert exc_info.value.budget_type == 'daily'
        assert exc_info.value.limit == 10.0

    def test_check_budget_exceeds_daily_without_enforce(self, tmp_path):
        """Test budget check returns False when exceeded with enforce=False."""
        tracker = BudgetTracker(
            daily_budget=10.0,
            enforce_limits=False,
            storage_dir=str(tmp_path)
        )
        tracker.add_cost(9.0)

        result = tracker.check_budget(2.0)
        assert result is False

    def test_check_budget_exceeds_session(self, tmp_path):
        """Test budget check with session budget."""
        tracker = BudgetTracker(
            daily_budget=20.0,
            session_budget=5.0,
            enforce_limits=True,
            storage_dir=str(tmp_path)
        )
        tracker.add_cost(4.0)

        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check_budget(2.0)

        assert exc_info.value.budget_type == 'session'
        assert exc_info.value.limit == 5.0


class TestBudgetPersistence:
    """Test budget data persistence."""

    def test_daily_spending_persists(self, tmp_path):
        """Test that daily spending is saved and loaded."""
        # Create tracker and add cost
        tracker1 = BudgetTracker(storage_dir=str(tmp_path))
        tracker1.add_cost(5.0)

        # Create new tracker instance - should load the saved data
        tracker2 = BudgetTracker(storage_dir=str(tmp_path))
        assert tracker2.daily_spent == 5.0

    def test_session_spending_resets(self, tmp_path):
        """Test that session spending resets for new instances."""
        tracker1 = BudgetTracker(storage_dir=str(tmp_path))
        tracker1.add_cost(5.0)

        tracker2 = BudgetTracker(storage_dir=str(tmp_path))
        assert tracker2.session_spent == 0.0  # Session resets
        assert tracker2.daily_spent == 5.0  # But daily persists

    def test_corrupted_budget_file(self, tmp_path):
        """Test handling of corrupted budget file."""
        # Create a corrupted file
        budget_file = tmp_path / "budget_tracker.json"
        with open(budget_file, 'w') as f:
            f.write("this is not valid json{{{")

        # Should handle gracefully and start fresh
        tracker = BudgetTracker(storage_dir=str(tmp_path))
        assert tracker.daily_spent == 0.0

    def test_old_date_resets_budget(self, tmp_path):
        """Test that budget resets for a new day."""
        # Create old budget file with yesterday's date
        budget_file = tmp_path / "budget_tracker.json"
        old_data = {
            'date': '2020-01-01',  # Old date
            'spent': 100.0,
            'last_updated': '2020-01-01T12:00:00'
        }
        with open(budget_file, 'w') as f:
            json.dump(old_data, f)

        # Should reset because date doesn't match today
        tracker = BudgetTracker(storage_dir=str(tmp_path))
        assert tracker.daily_spent == 0.0


class TestBudgetStatus:
    """Test budget status reporting."""

    def test_get_remaining_budget(self, tmp_path):
        """Test getting remaining budget."""
        tracker = BudgetTracker(
            daily_budget=10.0,
            session_budget=3.0,
            storage_dir=str(tmp_path)
        )
        tracker.add_cost(2.5)

        remaining = tracker.get_remaining_budget()
        assert remaining['daily_remaining'] == 7.5
        assert remaining['session_remaining'] == 0.5
        assert remaining['daily_spent'] == 2.5
        assert remaining['session_spent'] == 2.5

    def test_get_remaining_budget_no_session(self, tmp_path):
        """Test remaining budget without session budget."""
        tracker = BudgetTracker(
            daily_budget=10.0,
            storage_dir=str(tmp_path)
        )
        tracker.add_cost(3.0)

        remaining = tracker.get_remaining_budget()
        assert remaining['daily_remaining'] == 7.0
        assert remaining['session_remaining'] is None

    def test_is_near_limit_daily(self, tmp_path):
        """Test near limit detection for daily budget."""
        tracker = BudgetTracker(
            daily_budget=10.0,
            warning_threshold=0.8,
            storage_dir=str(tmp_path)
        )

        # Below threshold
        tracker.add_cost(7.0)
        assert tracker.is_near_limit() is False

        # At threshold
        tracker.add_cost(1.0)  # Now at 80%
        assert tracker.is_near_limit() is True

    def test_is_near_limit_session(self, tmp_path):
        """Test near limit detection for session budget."""
        tracker = BudgetTracker(
            daily_budget=20.0,
            session_budget=5.0,
            warning_threshold=0.8,
            storage_dir=str(tmp_path)
        )

        # Below threshold
        tracker.add_cost(3.0)
        assert tracker.is_near_limit() is False

        # At threshold (80% of 5.0 = 4.0)
        tracker.add_cost(1.0)
        assert tracker.is_near_limit() is True

    def test_reset_session(self, tmp_path):
        """Test resetting session budget."""
        tracker = BudgetTracker(
            daily_budget=20.0,
            session_budget=5.0,
            storage_dir=str(tmp_path)
        )

        tracker.add_cost(3.0)
        assert tracker.session_spent == 3.0
        assert tracker.daily_spent == 3.0

        tracker.reset_session()
        assert tracker.session_spent == 0.0
        assert tracker.daily_spent == 3.0  # Daily unchanged

    def test_get_status_report(self, tmp_path):
        """Test status report generation."""
        tracker = BudgetTracker(
            daily_budget=10.0,
            session_budget=3.0,
            storage_dir=str(tmp_path)
        )
        tracker.add_cost(2.0)

        report = tracker.get_status_report()
        assert "Daily Budget" in report
        assert "10.00" in report
        assert "2.0000" in report
        assert "Session Budget" in report
        assert "3.00" in report

    def test_status_report_near_limit(self, tmp_path):
        """Test status report includes warning when near limit."""
        tracker = BudgetTracker(
            daily_budget=10.0,
            warning_threshold=0.8,
            storage_dir=str(tmp_path)
        )
        tracker.add_cost(9.0)  # 90% of budget

        report = tracker.get_status_report()
        assert "WARNING" in report

    def test_repr(self, tmp_path):
        """Test string representation."""
        tracker = BudgetTracker(
            daily_budget=10.0,
            session_budget=5.0,
            storage_dir=str(tmp_path)
        )
        tracker.add_cost(2.5)

        repr_str = repr(tracker)
        assert "daily=$10.00" in repr_str
        assert "session=$5.0" in repr_str
        assert "daily_spent=$2.50" in repr_str
