"""CPU-only dependency, selection and interpreter regressions."""
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('evaluate_when_ready_tested', ROOT / 'scripts/evaluate_when_ready.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class EvaluateWhenReadyTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def completed_run(self, step=36000):
        (self.root / 'status.json').write_text(json.dumps({'status': 'completed', 'step': step, 'pid': 1}))
        checkpoint = self.root / 'checkpoints' / f'{step:06d}'
        checkpoint.mkdir(parents=True)
        for name in ('config.json', 'model.pt', 'normalization.json', 'train_config.json'):
            (checkpoint / name).touch()
        (checkpoint / 'checkpoint_complete.json').write_text(json.dumps({'step': step}))

    def test_completed_exact_budget_does_not_require_live_training_pid(self):
        self.completed_run()
        result = module.inspect_run(self.root, 36000, alive=lambda _: False)
        self.assertEqual(result['status'], 'completed')

    def test_failed_wrong_step_dead_pid_and_partial_checkpoint_rejected(self):
        self.completed_run(6000)
        with self.assertRaisesRegex(RuntimeError, 'expected 36000'):
            module.inspect_run(self.root, 36000)
        (self.root / 'failure.json').write_text('{"error":"OOM"}')
        with self.assertRaisesRegex(RuntimeError, 'failure recorded'):
            module.inspect_run(self.root, 6000)
        (self.root / 'failure.json').unlink()
        (self.root / 'status.json').write_text('{"status":"running","step":4000,"pid":123}')
        with self.assertRaisesRegex(RuntimeError, 'is dead'):
            module.inspect_run(self.root, 6000, alive=lambda _: False)
        (self.root / 'status.json').write_text('{"status":"completed","step":6000}')
        (self.root / 'checkpoints/006000/model.pt').unlink()
        with self.assertRaisesRegex(RuntimeError, 'complete final checkpoint'):
            module.inspect_run(self.root, 6000)

    def test_selection_deduplicates_symlink_and_preserves_role_mapping(self):
        checkpoint = self.root / '036000'
        checkpoint.mkdir()
        link = self.root / 'linked'
        link.symlink_to(checkpoint, target_is_directory=True)
        report = {'runs': [{'run': 'a', 'best_saved': {'step': 36000, 'checkpoint': str(link)},
                            'final_equal_budget': {'step': 36000, 'checkpoint': str(checkpoint)}}]}
        groups, selections = module.select_unique_checkpoints(report, 36000)
        self.assertEqual(groups, {'best_saved': [str(checkpoint)], 'final_equal_budget': []})
        self.assertEqual([s['selection'] for s in selections], ['best_saved', 'final_equal_budget'])
        self.assertEqual([s['evaluated_in'] for s in selections], ['best_saved', 'best_saved'])
        with self.assertRaises(ValueError):
            module.select_unique_checkpoints(report, 6000)

    def test_command_preserves_virtualenv_interpreter_string(self):
        executable = self.root / '.venv/bin/python'
        executable.parent.mkdir(parents=True)
        executable.symlink_to(Path(sys.executable).resolve())
        python = str(executable)
        self.assertNotEqual(python, str(executable.resolve()))
        with patch.object(module.sys, 'executable', python):
            command = module.python_command('evaluate_offline.py', ['--device', 'cuda:0'])
        self.assertEqual(command[0], python)
        self.assertEqual(command[-2:], ['--device', 'cuda:0'])

    def test_implementation_mutation_rejected(self):
        with patch.object(module, 'implementation_hashes', return_value={'a.py': 'new'}):
            with self.assertRaisesRegex(RuntimeError, 'implementation changed'):
                module.verify_implementation({'a.py': 'old'})


if __name__ == '__main__':
    unittest.main()
