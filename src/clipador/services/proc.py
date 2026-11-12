"""Process helpers."""

from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)

_job_handle = None  # keep the Windows job object handle alive for the process lifetime


def kill_children_on_exit() -> bool:
    """On Windows, put this process in a Job Object with KILL_ON_JOB_CLOSE.

    Child processes (yt-dlp, ffmpeg) inherit the job, so if the worker is killed (closing the
    terminal, Task Manager, crash) they die too instead of continuing as orphans that would
    clash with the job being retried. On other OSes this is a no-op (Ctrl+C reaches the whole
    process group there).
    """
    global _job_handle
    if sys.platform != "win32" or _job_handle is not None:
        return _job_handle is not None
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_uint64)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    job_object_extended_limit_information = 9
    job_object_limit_kill_on_job_close = 0x2000

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.GetCurrentProcess.restype = wintypes.HANDLE

    job = k32.CreateJobObjectW(None, None)
    if not job:
        log.warning("CreateJobObject falhou (erro %s)", ctypes.get_last_error())
        return False
    info = ExtendedLimits()
    info.BasicLimitInformation.LimitFlags = job_object_limit_kill_on_job_close
    ok = k32.SetInformationJobObject(
        job, job_object_extended_limit_information, ctypes.byref(info), ctypes.sizeof(info)
    )
    if not ok or not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        log.warning(
            "Não foi possível vincular processos filhos ao worker (erro %s); "
            "se o worker cair, confira se sobrou algum yt-dlp/ffmpeg rodando.",
            ctypes.get_last_error(),
        )
        return False
    _job_handle = job
    return True
