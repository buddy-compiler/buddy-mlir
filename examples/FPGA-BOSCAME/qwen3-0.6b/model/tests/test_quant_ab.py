import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimization"))
from archive_quant_ab import parse


class QuantEvidence(unittest.TestCase):
    def test_complete_and_rejected_records(self):
        manifest = {"host": False, "shape": [1, 1024], "repeats": 2}
        lines = ["[quant-ab] rows=00000001 cols=00000400"]
        for i, v in enumerate(("baseline", "optimized", "optimized", "baseline")):
            lines.append(f"[quant-ab] round={i//2:08X} variant={v} cycles=0000000000000100 errors=00000000")
        lines += ["[quant-ab] PASS errors=00000000", "[nr] RA returned: PASS", "verify NR runtime: PASS"]
        text = "\n".join(lines)
        self.assertEqual(parse(text, platform="fpga", manifest=manifest)["timing"]["mean_speedup"], 1)
        for invalid in ("\n".join(lines[:2]+lines[3:]), text+"\n"+lines[1],
                        text.replace("errors=00000000", "errors=00000001", 1),
                        text.replace("cycles=0000000000000100", "cycles=0000000000000000", 1),
                        text.replace("variant=baseline", "variant=optimized", 1),
                        text.replace("[nr] RA returned: PASS", "")):
            with self.assertRaises(ValueError):
                parse(invalid, platform="fpga", manifest=manifest)


if __name__ == "__main__":
    unittest.main()
