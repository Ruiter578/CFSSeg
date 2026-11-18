import os
import shlex
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RunnerContractTests(unittest.TestCase):
    def run_script(self, script='run.sh', **overrides):
        env = dict(os.environ)
        for key in ['MODEL', 'SUBPATH', 'BASE_SUBPATH', 'AIR_FEATURE_SOURCE', 'DATA_ROOT',
                    'START_STEP', 'END_STEP', 'STEP_INCREMENT', 'PRETRAINED_BACKBONE', 'TMPDIR']:
            env.pop(key, None)
        env.update(PYTHON='/bin/echo', START_STEP='0', END_STEP='1',
                   SUBPATH='runner_contract', PRETRAINED_BACKBONE='')
        env.update(overrides)
        return subprocess.run(['bash', str(ROOT / script)], cwd='/tmp', env=env,
                              capture_output=True, text=True, timeout=5)

    def commands(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('train.py ')]

    def test_default_and_optional_model_keep_their_own_step0(self):
        for model in ['deeplabv3_resnet101', 'deeplabv3plus_resnet101']:
            with self.subTest(model=model):
                commands = self.commands(self.run_script(MODEL=model))
                self.assertEqual(len(commands), 2)
                for args in commands:
                    self.assertEqual(args[args.index('--model') + 1], model)
                    self.assertEqual(args[args.index('--air_feature_source') + 1], 'auto')
                    self.assertEqual(args[args.index('--base_subpath') + 1], 'runner_contract')
                    self.assertEqual(args[args.index('--data_root') + 1], str(ROOT / 'data_root/VOC2012'))
                    self.assertNotIn('--pretrained_backbone', args)

    def test_explicit_existing_base_is_preserved(self):
        commands = self.commands(self.run_script(START_STEP='1', END_STEP='1', BASE_SUBPATH='existing_v3plus_base'))
        self.assertEqual(commands[0][commands[0].index('--base_subpath') + 1], 'existing_v3plus_base')

    def test_trs_and_rhl_wrappers_forward_model_and_new_protocol_options(self):
        for script in ['run_trs.sh', 'run_rhl_norm.sh']:
            with self.subTest(script=script):
                result = self.run_script(script, MODEL='deeplabv3plus_resnet101',
                                         AIR_FEATURE_SOURCE='aspp_up', EVALUATION_MODE='val',
                                         TRAIN_EXCLUDE_LIST='/tmp/exclusion.txt', VALIDATION_LIST='/tmp/holdout.txt',
                                         USE_LOCAL_MPS='0')
                args = self.commands(result)[0]
                for key, value in [('--model', 'deeplabv3plus_resnet101'), ('--air_feature_source', 'aspp_up'),
                                   ('--evaluation_mode', 'val'), ('--train_exclude_list', '/tmp/exclusion.txt'),
                                   ('--validation_list', '/tmp/holdout.txt')]:
                    self.assertIn(key, args)
                    self.assertEqual(args[args.index(key) + 1], value)

    def test_zero_step_increment_stops_before_launching(self):
        result = self.run_script(STEP_INCREMENT='0', START_STEP='1', END_STEP='0')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('STEP_INCREMENT', result.stderr)


if __name__ == '__main__':
    unittest.main()
