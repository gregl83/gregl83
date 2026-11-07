import argparse
from pathlib import Path
import tempfile
import unittest

from scripts.preview_showcase import output_directory, ROOT, write_preview


class ReadinessTests(unittest.TestCase):
    def test_preview_cannot_target_tracked_paths(self):
        for path in (ROOT, ROOT / 'assets/profile', ROOT / 'out', ROOT / 'out/../assets'):
            with self.assertRaises(argparse.ArgumentTypeError):
                output_directory(path)
        self.assertEqual(output_directory(ROOT / 'out/test'), ROOT / 'out/test')

    def test_local_preview_links_are_self_contained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_preview(root)
            html = (root / 'preview.html').read_text()
            self.assertIn('srcset="intro-mobile.gif"', html)
            self.assertNotIn('releases/latest', html)
            readme = (root / 'README.md').read_text()
            self.assertIn('srcset="intro-mobile.gif"', readme)
            self.assertIn('href="release-notes.md"', readme)
            self.assertNotIn('https://', readme)
            write_preview(root, example=True)
            self.assertNotIn('href="showcase.json"', (root / 'preview.html').read_text())
            self.assertEqual((ROOT / 'assets/profile/README.md').read_text(), (root / 'README.md').read_text())
            self.assertTrue((root / 'showcase-example.md').is_file())


if __name__ == '__main__':
    unittest.main()
