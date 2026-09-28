"""Compiled export must include the source joint-limit parameter arrays."""

import unittest

from bevy_microduck_tools.model_export import FIELDS


class ModelExportLimitFieldTests(unittest.TestCase):
    def test_source_joint_limit_parameters_are_exported_once(self):
        limited = FIELDS.index("jnt_limited")
        self.assertEqual(
            FIELDS[limited + 1:limited + 4],
            ("jnt_solref", "jnt_solimp", "jnt_margin"),
        )
        for name in ("jnt_solref", "jnt_solimp", "jnt_margin", "dof_invweight0"):
            self.assertEqual(FIELDS.count(name), 1)
