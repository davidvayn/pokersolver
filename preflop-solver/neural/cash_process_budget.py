"""POSIX cash research process budgets, including timing-wrapper descendants.

Commands must stay in their spawned session: detached/daemonized commands are
not supported. No shell, credentials, process-name scanning or broad PID kills.
Only the newly created process group can be terminated. Completed calls retain
the ordinary CompletedProcess/check_returncode interface and captured logs.
"""
import math
import os
import signal
import subprocess


def run_cash_process(args,*,timeout,env=None,termination_grace_seconds=.5):
    if (os.name!='posix' or not isinstance(args,(list,tuple)) or not args
            or any(not isinstance(arg,(str,os.PathLike)) for arg in args)
            or type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=0
            or type(termination_grace_seconds) not in (int,float)
            or not math.isfinite(termination_grace_seconds)
            or not 0<termination_grace_seconds<=5):
        raise ValueError('Expected a POSIX command and finite positive process budgets')
    process=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
        env=env,start_new_session=True)

    def stop_group(which):
        try:
            os.killpg(process.pid,which)
        except ProcessLookupError:
            pass

    try:
        stdout,stderr=process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as original:
        stop_group(signal.SIGTERM)
        try:
            stdout,stderr=process.communicate(timeout=termination_grace_seconds)
        except subprocess.TimeoutExpired:
            stop_group(signal.SIGKILL)
            stdout,stderr=process.communicate()
        # A descendant may close/redirect inherited log pipes and ignore TERM;
        # successful wrapper communication alone does not prove it stopped.
        stop_group(signal.SIGKILL)
        error=subprocess.TimeoutExpired(args,timeout,output=stdout,stderr=stderr)
        # Explicit identity aids diagnostic callers without searching names or
        # touching other jobs. The wrapper is reaped; no group member can
        # retain live compute (the OS may briefly retain an orphaned zombie).
        error.process_group_id=process.pid
        raise error from original
    except BaseException:
        stop_group(signal.SIGKILL)
        process.wait()
        raise
    return subprocess.CompletedProcess(args,process.returncode,stdout,stderr)
