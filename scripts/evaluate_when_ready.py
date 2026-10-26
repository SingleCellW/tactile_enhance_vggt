#!/usr/bin/env python3
"""One-shot, local background dependency evaluator; never launches/stops training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile_vggt.execution import selected_gpus, child_environment


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def implementation_hashes():
    paths = [ROOT / 'scripts' / name for name in
             ('evaluate_when_ready.py', 'evaluate_offline.py', 'evaluate_sim.py', 'sim_worker.py', 'prepare_cache.py')]
    paths += sorted((ROOT / 'src/tactile_vggt').rglob('*.py'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def verify_implementation(expected):
    if implementation_hashes() != expected:
        raise RuntimeError('Evaluation implementation changed after submission; create a new evaluation job')


def process_alive(pid):
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        stat = Path(f'/proc/{pid}/stat')
        if stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
            return False
        return True
    except (ValueError, TypeError, ProcessLookupError, FileNotFoundError):
        return False
    except PermissionError:
        return True


def inspect_run(run, expected_steps, alive=process_alive):
    """Return a dependency snapshot, rejecting failures and terminal wrong budgets."""
    run = Path(run)
    failure = run / 'failure.json'
    if failure.exists():
        raise RuntimeError(f'Training failure recorded at {failure}: {failure.read_text()[:1000]}')
    path = run / 'status.json'
    if not path.exists():
        return {'run': str(run), 'status': 'waiting_for_status'}
    status = json.loads(path.read_text())
    state, step = status.get('status'), status.get('step')
    if state == 'failed':
        raise RuntimeError(f'Training failed: {run}')
    if state == 'completed':
        if step != expected_steps:
            raise RuntimeError(f'Completed run has step {step}, expected {expected_steps}: {run}')
        checkpoint = run / 'checkpoints' / f'{expected_steps:06d}'
        required = ('checkpoint_complete.json', 'config.json', 'model.pt', 'normalization.json', 'train_config.json')
        if not all((checkpoint / name).is_file() for name in required):
            raise RuntimeError(f'Completed run lacks a complete final checkpoint: {checkpoint}')
        if json.loads((checkpoint / 'checkpoint_complete.json').read_text()).get('step') != expected_steps:
            raise RuntimeError(f'Final checkpoint marker has the wrong step: {checkpoint}')
    elif status.get('pid') is not None and not alive(status['pid']):
        raise RuntimeError(f'Training process {status["pid"]} is dead: {run}')
    elif isinstance(step, int) and step > expected_steps:
        raise RuntimeError(f'Training exceeded the requested budget: {run}')
    return {'run': str(run), 'status': state, 'step': step, 'pid': status.get('pid')}


def select_unique_checkpoints(report, expected_steps):
    """Run each physical checkpoint once, retaining both scientific selection labels."""
    groups = {'best_saved': [], 'final_equal_budget': []}
    selections, owner = [], {}
    for role in groups:
        for run in report['runs']:
            selected = run.get(role)
            if not selected or (role == 'final_equal_budget' and selected['step'] != expected_steps):
                raise ValueError(f'Missing/wrong {role} checkpoint for {run["run"]}')
            path = str(Path(selected['checkpoint']).resolve())
            if path not in owner:
                owner[path] = role
                groups[role].append(path)
            selections.append({'run': run['run'], 'selection': role, 'step': selected['step'],
                               'checkpoint': path, 'evaluated_in': owner[path]})
    return groups, selections


def python_command(script, arguments, python=None):
    # Resolving the executable symlink would silently escape a virtual environment.
    return [sys.executable if python is None else python, str(ROOT / 'scripts' / script), *arguments]


def stop_children(children):
    for child, _ in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for child, _ in children:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def execute_jobs(jobs, expected_hashes, deadline, output):
    children = []
    try:
        for name, gpu, command in jobs:
            verify_implementation(expected_hashes)
            if time.monotonic() >= deadline:
                raise TimeoutError('Evaluation job timeout expired')
            env = child_environment(gpu, render=True)
            log = (output / f'{name}.log').open('x')
            try:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         cwd=ROOT, start_new_session=True)
            except BaseException:
                log.close()
                raise
            children.append((child, log))
        while any(child.poll() is None for child, _ in children):
            if time.monotonic() >= deadline:
                raise TimeoutError('Evaluation job timeout expired')
            for child, _ in children:
                if child.poll() not in (None, 0):
                    raise RuntimeError(f'Evaluation subprocess {child.pid} failed with exit {child.returncode}; inspect logs')
            time.sleep(min(2, max(.01, deadline - time.monotonic())))
        if any(child.returncode != 0 for child, _ in children):
            raise RuntimeError('Evaluation subprocess failed; inspect logs')
    finally:
        stop_children(children)
        for _, log in children:
            log.close()


def run(args):
    gpus = selected_gpus(args.gpus)
    if len(gpus) != 2 or len(set(gpus)) != 2 or not all(gpus):
        raise ValueError('--gpus requires exactly two distinct physical GPU IDs/UUIDs')
    if args.expected_steps < 1 or args.timeout_seconds <= 0 or args.poll_seconds <= 0:
        raise ValueError('Steps, timeout and poll interval must be positive')
    runs = [str(Path(path).resolve()) for path in args.runs]
    if not runs or len(runs) != len(set(runs)):
        raise ValueError('Supply unique training run directories')
    cache = Path(args.cache).resolve(strict=True)
    manifest = json.loads((cache / 'manifest.json').read_text())
    if (not manifest.get('complete') or manifest.get('native_tactile_report', {}).get('velocity_dt') != .01
            or manifest.get('configuration', {}).get('action_label_repair', {}).get('action_target_offset_frames') != 0):
        raise ValueError('This evaluator requires the physics-aligned v2 cache (dt=.01, current command)')
    episodes = args.episodes or ','.join(map(str, manifest['split']['val']))
    output = Path(args.output).absolute()
    # Never reuse even an empty output directory or touch an existing evaluation.
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    deadline = time.monotonic() + args.timeout_seconds
    hashes = implementation_hashes()
    plan = {'runs': runs, 'cache': str(cache), 'python': sys.executable, 'gpus': gpus,
            'cache_manifest_sha256': hashlib.sha256((cache / 'manifest.json').read_bytes()).hexdigest(),
            'expected_steps': args.expected_steps, 'timeout_seconds': args.timeout_seconds,
            'episodes': episodes, 'max_steps': 600, 'sensor_dt': .01,
            'implementation_sha256': hashes, 'created_at': started,
            'selection_caveat': 'Validation-selected best is a development result; final is the equal-budget comparison.'}
    write_json(output / 'plan.json', plan)

    def status(state, **extra):
        write_json(output / 'status.json', {'status': state, 'pid': os.getpid(), 'started_at': started,
                   'updated_at': datetime.now(timezone.utc).isoformat(), **extra})

    try:
        while True:
            snapshots = [inspect_run(path, args.expected_steps) for path in runs]
            status('waiting', dependencies=snapshots)
            if all(item['status'] == 'completed' for item in snapshots):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Timed out waiting for training dependencies')
            time.sleep(min(args.poll_seconds, max(.01, deadline - time.monotonic())))
        status('training_done', dependencies=snapshots)
        if hashlib.sha256((cache / 'manifest.json').read_bytes()).hexdigest() != plan['cache_manifest_sha256']:
            raise RuntimeError('Training cache manifest changed after evaluation submission')
        offline = output / 'offline.json'
        command = python_command('evaluate_offline.py', ['--runs', *runs, '--cache', str(cache),
                                  '--output', str(offline), '--device', 'cuda:0'])
        status('offline_evaluation', command=command)
        execute_jobs([('offline', gpus[0], command)], hashes, deadline, output)
        report = json.loads(offline.read_text())
        if {str(Path(item['run']).resolve()) for item in report['runs']} != set(runs) or len(report['runs']) != len(runs):
            raise RuntimeError('Offline report did not evaluate every requested run exactly once')
        groups, selections = select_unique_checkpoints(report, args.expected_steps)
        write_json(output / 'selection.json', {'groups': groups, 'selections': selections})
        jobs = []
        for (role, checkpoints), gpu in zip(groups.items(), gpus):
            if not checkpoints:
                continue
            arguments = [item for path in checkpoints for item in ('--checkpoint', path)]
            arguments += ['--cache', str(cache), '--output', str(output / role), '--device', 'cuda:0',
                          '--episodes', episodes, '--max-steps', '600', '--substeps', '1', '--sensor-dt', '.01',
                          '--video', '--selection-note', role + '; fixed offline selection before closed-loop evaluation']
            if args.worker_python:
                arguments += ['--worker-python', args.worker_python]
            if args.vggt_weights:
                arguments += ['--vggt-weights', args.vggt_weights]
            jobs.append((role, gpu, python_command('evaluate_sim.py', arguments)))
        status('closed_loop', jobs=[{'selection': name, 'gpu': gpu, 'command': cmd} for name, gpu, cmd in jobs])
        execute_jobs(jobs, hashes, deadline, output)
        for role, _, _ in jobs:
            if not json.loads((output / role / 'summary.json').read_text()).get('complete'):
                raise RuntimeError(f'Closed-loop evaluation is incomplete: {role}')
        status('completed', offline=str(offline), selections=str(output / 'selection.json'),
               closed_loop_outputs=[str(output / name) for name, _, _ in jobs])
    except BaseException as error:
        failure = {'error': str(error), 'traceback': traceback.format_exc()}
        write_json(output / 'failure.json', failure)
        status('failed', **failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpus', required=True, help='Two distinct allocated GPU IDs/UUIDs')
    parser.add_argument('--expected-steps', type=int, default=36000)
    parser.add_argument('--timeout-seconds', type=float, default=48 * 3600)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--episodes', help='Comma-separated validation IDs; default all five held-out episodes')
    parser.add_argument('--worker-python')
    parser.add_argument('--vggt-weights')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
