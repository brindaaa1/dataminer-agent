"""建模子进程的启动、中止与存活检查。第一版每个任务一个子进程，web 服务记着 Popen 对象；
以后换成任务队列或容器沙箱，只改这个文件。"""
import os
import signal
import subprocess
import sys

from data.sample import ROOT
from server import tasks

_procs: dict[str, subprocess.Popen] = {}


def start(tid: str):
    tasks.update(tid, status="running", error=None)     # 先改状态再起进程：进程跑得快时，它写的 done 不会被这里覆盖
    log = open(tasks.task_dir(tid) / "worker.log", "ab")
    _procs[tid] = subprocess.Popen([sys.executable, "-m", "server.worker", tid], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT)


def alive(tid: str, pid: int | None) -> bool:
    p = _procs.get(tid)
    if p is not None:
        return p.poll() is None                         # 自己起的进程用 poll：被杀后成了僵尸进程，kill(pid, 0) 仍会成功
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def stop(tid: str):
    p = _procs.pop(tid, None)
    if p is not None:
        p.terminate()
        p.wait(10)
    elif pid := tasks.load(tid).get("pid"):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    tasks.update(tid, status="stopped", pid=None)


def refresh(meta: dict) -> dict:
    """状态是"运行中"但进程已经不在（崩溃、被杀、web 服务重启前启动的）→ 标为失败，可以续跑。
    接入同理：标着"起草中"但本进程里没有对应线程 → 接入失败，可以重试。"""
    from server import intake_runner
    meta = intake_runner.refresh(meta)
    if meta["status"] == "running" and not alive(meta["id"], meta.get("pid")):
        return tasks.update(meta["id"], status="failed", error="建模进程意外退出，详见任务目录下的 worker.log", pid=None)
    return meta
