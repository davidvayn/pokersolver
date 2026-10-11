import os
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock,patch

from cash_process_budget import run_cash_process


def timed_command(arguments):
    # Resource flag -l is macOS-only; the process-group failure pattern also
    # needs to be covered by GNU time on Linux rather than failing on its CLI.
    return ['/usr/bin/time']+(['-l'] if sys.platform=='darwin' else [])+arguments


@unittest.skipUnless(os.name=='posix','cash research budgets require POSIX sessions')
class CashProcessBudgetTests(unittest.TestCase):
    def test_success_and_nonzero_completion_keep_normal_captured_interface(self):
        result=run_cash_process([sys.executable,'-c','import sys; print("ok"); print("diagnostic",file=sys.stderr)'],timeout=2)
        self.assertEqual(result.returncode,0); self.assertEqual(result.stdout,'ok\n')
        self.assertEqual(result.stderr,'diagnostic\n'); result.check_returncode()
        result=run_cash_process([sys.executable,'-c','import sys; sys.exit(7)'],timeout=2)
        self.assertEqual(result.returncode,7)
        with self.assertRaises(subprocess.CalledProcessError): result.check_returncode()

    @unittest.skipUnless(os.path.isfile('/usr/bin/time'),'timing wrapper unavailable')
    def test_actual_time_wrapper_and_term_ignoring_solver_both_stop_on_timeout(self):
        code='import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print(os.getpgrp(),os.getpid(),flush=True); time.sleep(20)'
        started=time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired) as caught:
            run_cash_process(timed_command([sys.executable,'-u','-c',code]),
                timeout=.2,termination_grace_seconds=.05)
        error=caught.exception; self.assertIsInstance(error.stdout,str)
        group,child=map(int,error.stdout.splitlines()[0].split())
        self.assertEqual(group,error.process_group_id); self.assertNotEqual(group,os.getpgrp())
        self.assertLess(time.monotonic()-started,2.)
        for pid in (group,child):
            # A short-lived reparented zombie can await the OS reaper, but it
            # must not retain a live solver or any compute after the budget.
            stat=subprocess.run(['ps','-o','stat=','-p',str(pid)],capture_output=True,text=True).stdout.strip()
            self.assertTrue(not stat or stat.startswith('Z'),f'budget left live PID {pid}: {stat}')


    @unittest.skipUnless(os.path.isfile('/usr/bin/time'),'timing wrapper unavailable')
    def test_redirected_background_child_cannot_survive_wrapper_cleanup(self):
        child_code='import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready",flush=True); time.sleep(20)'
        code=('import os,subprocess,sys,time; child=subprocess.Popen([sys.executable,"-u","-c",'
            +repr(child_code)+'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL); child.stdout.readline(); '
            'print(os.getpgrp(),child.pid,flush=True); time.sleep(20)')
        with self.assertRaises(subprocess.TimeoutExpired) as caught:
            run_cash_process(timed_command([sys.executable,'-u','-c',code]),
                timeout=.2,termination_grace_seconds=.05)
        error=caught.exception; group,child=map(int,error.stdout.splitlines()[0].split())
        try:
            stat=subprocess.run(['ps','-o','stat=','-p',str(child)],capture_output=True,text=True).stdout.strip()
            self.assertTrue(not stat or stat.startswith('Z'),f'closed pipes concealed live child {child}: {stat}')
        finally:
            # Even a failed red regression cleans up only its own fixture.
            try: os.killpg(group,signal.SIGKILL)
            except ProcessLookupError: pass

    def test_bad_budget_or_command_fails_before_spawning(self):
        with patch('cash_process_budget.subprocess.Popen') as spawn:
            for timeout in (0.,-1.,float('inf'),float('nan'),True):
                with self.assertRaises(ValueError): run_cash_process(['nothing'],timeout=timeout)
            for grace in (0.,-1.,6.,True,float('inf')):
                with self.assertRaises(ValueError): run_cash_process(['nothing'],timeout=1,termination_grace_seconds=grace)
            for args in ('nothing',[],[None]):
                with self.assertRaises(ValueError): run_cash_process(args,timeout=1)
            spawn.assert_not_called()

    def test_unexpected_exception_kills_only_its_own_group_and_reaps(self):
        process=Mock(pid=123456); process.communicate.side_effect=RuntimeError('fixture')
        with patch('cash_process_budget.subprocess.Popen',return_value=process) as spawn:
            with patch('cash_process_budget.os.killpg') as kill:
                with self.assertRaisesRegex(RuntimeError,'fixture'): run_cash_process(['fixture'],timeout=1)
                kill.assert_called_once_with(123456,signal.SIGKILL); process.wait.assert_called_once()
                self.assertTrue(spawn.call_args.kwargs['start_new_session'])


if __name__=='__main__': unittest.main()
