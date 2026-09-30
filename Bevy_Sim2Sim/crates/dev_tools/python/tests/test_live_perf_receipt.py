"""CPU receipt checks for a v3 live preview report. No window or GPU."""

import unittest

from bevy_microduck_tools.live_perf_receipt import analyze


def _counter() -> dict:
    return {
        "step_ns": 5,
        "broad_phase_ns": 1,
        "final_broad_phase_ns": 1,
        "narrow_phase_ns": 2,
        "islands_ns": 3,
        "constraints_ns": 4,
        "solver_ns": 10,
        "ccd_ns": 0,
        "user_changes_ns": 0,
    }


def _report(*, ticks: int = 4, rapier: str = "compiled",
            frame_count: int = 4, frame_sum_ns: int = 80_000_000) -> dict:
    report = {
        "schema": "station_robot_live_preview_v3",
        "minimum_ticks": ticks,
        "actual_completed_ticks": ticks,
        "policy_inference_count": ticks,
        "integration_count": ticks,
        "completed_publishes": ticks,
        "last_displayed_step": ticks,
        "trace": [{"tick_before": step} for step in range(ticks)],
        "pose_frames": [{"global_step": step} for step in range(ticks + 1)],
        "final_clock": {"global_step": ticks, "pending_ticks": 0},
        "worker_finished": True,
        "worker_wall_ns": 80_000_000,
        "worker_deadline_missed_ticks": 2,
        "worker_max_deadline_lag_ns": 2_000_000,
        "worker_60hz_deadline_met": False,
        "pending_ticks": 0,
        "passed": True,
        "performance_qualified": False,
        "skill_qualified": False,
    }
    if rapier == "absent":
        return report
    compiled = rapier == "compiled"
    report["live_profile"] = {
        "frame_start_to_start": {
            "count": frame_count, "sum_ns": frame_sum_ns, "max_ns": frame_sum_ns,
        },
        "advance_frame_cpu": {
            "count": ticks, "sum_ns": ticks * 1_000_000, "max_ns": 1_500_000,
        },
        "worker_deadline_lag": {"count": ticks, "sum_ns": 3_000_000, "max_ns": 2_000_000},
        "deadline_missed_ticks": 2,
        "rapier_profiler_compiled": compiled,
        "rapier_counters_per_tick": [_counter() for _ in range(ticks)] if compiled else [],
    }
    return report


class LivePerfReceiptTests(unittest.TestCase):
    def sampled_report(self):
        report = _report()
        report["display_frames"] = 5
        profile = report["live_profile"]
        profile["frame_start_to_start"]["max_ns"] = 20_000_000
        profile["frame_samples"] = [
            {"display_frame": i, "start_since_run_ns": i * 20_000_000,
             "previous_interval_ns": 20_000_000 if i else None}
            for i in range(5)
        ]
        profile["worker_tick_samples"] = [
            {"global_step": i + 1, "advance_cpu_ns": cpu, "deadline_lag_ns": lag}
            for i, (cpu, lag) in enumerate(zip(
                (500_000, 1_000_000, 1_000_000, 1_500_000), (0, 1_000_000, 0, 2_000_000)))
        ]
        for row in profile["rapier_counters_per_tick"]:
            row.update(ccd_toi_ns=7, update_ns=8, collision_detection_ns=9)
        return report

    def test_per_frame_and_native_ccd_samples_are_preserved(self):
        receipt = analyze(self.sampled_report())
        self.assertEqual(receipt["frame_intervals"]["count"], 4)
        self.assertEqual(receipt["frame_intervals"]["p95_ns"], 20_000_000)
        self.assertEqual(receipt["worker_tick_wall"]["p99_ns"], 1_500_000)
        self.assertEqual(receipt["rapier_counters"]["stages"]["ccd_toi_ns"]["sum_ns"], 28)

    def test_changed_tick_deadlines_are_rejected(self):
        report = self.sampled_report()
        report["live_profile"]["worker_tick_samples"][1]["deadline_lag_ns"] = 0
        with self.assertRaisesRegex(ValueError, "deadline samples disagree"):
            analyze(report)

    def test_changed_frame_intervals_are_rejected(self):
        report = self.sampled_report()
        report["live_profile"]["frame_samples"][2]["previous_interval_ns"] += 1
        with self.assertRaisesRegex(ValueError, "frame start timestamps"):
            analyze(report)

    def test_valid_report_stays_unqualified(self):
        receipt = analyze(_report())
        self.assertTrue(receipt["source_passed"])
        self.assertEqual(receipt["schema"], "station_robot_live_perf_receipt_v2")
        self.assertIn("wall_time", receipt["worker_advance_timing_semantics"])
        self.assertNotIn("worker_tick_cpu", receipt)
        self.assertNotIn("advance_frame_cpu", receipt)
        self.assertEqual(receipt["passed_means"], "wiring_only")
        self.assertFalse(receipt["performance_qualified"])
        self.assertFalse(receipt["skill_qualified"])
        self.assertEqual(receipt["physics_throughput_hz"], 4 * 1_000_000_000 / 80_000_000)
        self.assertEqual(receipt["advance_frame_wall"]["mean_ns"], 1_000_000)
        self.assertEqual(receipt["advance_frame_wall"]["max_ns"], 1_500_000)
        self.assertEqual(receipt["worker_deadline_lag"]["count"], 4)
        self.assertEqual(receipt["worker_deadline_lag"]["max_ns"], 2_000_000)
        self.assertEqual(receipt["worker_deadline_lag"]["over_zero_count"], 2)
        self.assertEqual(receipt["worker_deadline_missed_ticks"], 2)
        self.assertEqual(receipt["rapier_counters"]["status"], "compiled")
        self.assertEqual(receipt["rapier_counters"]["row_count"], 4)
        self.assertEqual(
            receipt["rapier_counters"]["stages"]["solver_ns"],
            {"sum_ns": 40, "max_ns": 10},
        )

    def test_missing_steps_are_rejected(self):
        report = _report()
        report["actual_completed_ticks"] = report["minimum_ticks"] - 1
        with self.assertRaisesRegex(ValueError, "actual_completed_ticks"):
            analyze(report)

    def test_trace_gap_is_rejected(self):
        report = _report()
        report["trace"][2]["tick_before"] = 4
        with self.assertRaisesRegex(ValueError, "trace"):
            analyze(report)

    def test_uncompiled_profiler_is_unavailable(self):
        receipt = analyze(_report(rapier="unavailable"))
        self.assertEqual(receipt["rapier_counters"], {
            "status": "unavailable",
            "reason": "rapier_profiler_not_compiled",
        })
        self.assertFalse(receipt["performance_qualified"])
        self.assertFalse(receipt["skill_qualified"])

    def test_display_fps_uses_frame_interval_sum(self):
        receipt = analyze(_report(frame_count=7, frame_sum_ns=200_000_000))
        self.assertEqual(receipt["display_fps"], 7 * 1_000_000_000 / 200_000_000)


if __name__ == "__main__":
    unittest.main()
