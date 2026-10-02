"""Regress the actual source warehouse graph that previously became gray."""

import os
from pathlib import Path
import unittest

from pxr import Usd, UsdShade

from unitree_g1_background_visual_export import surface, WAREHOUSE_MDL_SHA


class SourceWarehouseMaterial(unittest.TestCase):
    def setUp(self):
        value = os.environ.get("G1_BACKGROUND_MATERIAL_TEST_USD")
        if not value:
            self.skipTest("requires the original byte-bound USD/material/texture cache")
        self.usd = Path(value)
        self.stage = Usd.Stage.Open(str(self.usd))
        self.stage.SetEditTarget(self.stage.GetSessionLayer())
        self.mesh = self.stage.GetPrimAtPath(
            "/Lab/BackgroundAssets/boxes/hesai_box_06/SM_CardBoxC_02/SM_CardBoxC_02")
        material = UsdShade.MaterialBindingAPI(self.mesh).ComputeBoundMaterial()[0]
        self.shader = next(p for p in Usd.PrimRange(material.GetPrim()) if p.IsA(UsdShade.Shader))

    def test_actual_cardboard_retains_original_textures_tint_and_graph(self):
        mapped = surface(self.mesh, self.usd)
        self.assertEqual(mapped["base_color"], [0.5, 0.4000000059604645, 0.25])
        self.assertEqual(mapped["warehouse_mdl"]["source_mdl_sha256"], WAREHOUSE_MDL_SHA)
        self.assertEqual(mapped["warehouse_mdl"]["albedo_desaturation"], 0.5)
        for name, filename in (("albedo", "T_CardBoxC_D.png"),
                               ("normal", "T_CardBoxC_N.png"), ("orm", "T_CardBoxC_ORM.png")):
            self.assertEqual(Path(mapped[name]["path"]).name, filename)
            self.assertEqual(len(mapped[name]["sha256"]), 64)
        self.assertNotEqual(mapped["base_color"], [0.5, 0.5, 0.5])

    def test_changed_mdl_entry_is_rejected_instead_of_falling_back_to_gray(self):
        self.shader.GetAttribute("info:mdl:sourceAsset:subIdentifier").Set("changed_entry")
        with self.assertRaisesRegex(ValueError, "graph identity changed"):
            surface(self.mesh, self.usd)

    def test_nonfinite_desaturation_is_rejected_before_visual_export(self):
        UsdShade.Shader(self.shader).GetInput("Desaturation").Set(float("nan"))
        with self.assertRaisesRegex(ValueError, "finite mapping"):
            surface(self.mesh, self.usd)


if __name__ == "__main__":
    unittest.main()
