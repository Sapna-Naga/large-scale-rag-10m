"""Run a command with its console output and typed input appended to a daily IST log."""
import argparse
import datetime as dt
import os
import subprocess
import sys
import threading
from pathlib import Path

IST = dt.timezone(dt.timedelta(hours=5, minutes=30), 'IST')

class DailyLog:
    def __init__(self, directory='logs'):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def write(self, kind, message):
        now = dt.datetime.now(IST)
        # Fixed English month names keep filenames independent of OS locale.
        month = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')[now.month-1]
        name = f'{now.day:02d}{month}{now.year}.log'
        with self.lock, (self.directory / name).open('a', encoding='utf-8') as file:
            file.write(f'[{now.isoformat(timespec="seconds")} IST] [{kind}] {message.rstrip()}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', default='logs')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    # if not command:
    #     parser.error('Provide a command after --')
    # log = DailyLog(args.log_dir)
    if not command:
        parser.error('Provide a command after --')

    # Use the same Python interpreter that runs this logging script.
    # This prevents Windows from launching the global Python installation.
    if command[0].lower() in ("python", "python.exe", "python3"):
        command[0] = sys.executable

    log = DailyLog(args.log_dir)
    log.write('COMMAND', repr(command))
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=env, text=True, encoding='utf-8', errors='replace')
    except OSError as exc:
        log.write('ERROR', str(exc))
        raise

    def forward_input():
        try:
            for line in sys.stdin:
                log.write('INPUT', line)
                process.stdin.write(line)
                process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            try:
                process.stdin.close()
            except (OSError, ValueError):
                pass

    threading.Thread(target=forward_input, daemon=True).start()
    try:
        # Read characters so prompts without a newline still appear immediately.
        buffer = ''
        while char := process.stdout.read(1):
            sys.stdout.write(char)
            sys.stdout.flush()
            buffer += char
            if char in '\n\r' or len(buffer) >= 8192:
                log.write('OUTPUT', buffer)
                buffer = ''
        if buffer:
            log.write('OUTPUT', buffer)
        code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        code = process.wait()
    finally:
        log.write('EXIT', str(process.poll()))
    return code

if __name__ == '__main__':
    raise SystemExit(main())
