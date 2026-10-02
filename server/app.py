"""工作台后端：只做编排。用户操作走 POST，状态变化走事件流（GET .../stream，SSE）。

    uvicorn server.app:app --reload        # 开发时前端另起 vite（web/），/api 代理到这里
    uvicorn server.app:app                  # web/dist 存在时同时托管前端
"""
import asyncio
import io
import json
import os
import zipfile

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from data.sample import ROOT

os.chdir(ROOT)                                           # 与 app.py 一致：config.yaml、knowhow/ 等相对路径都以仓库根目录为准

from runtime.env import load_env  # noqa: E402
from runtime.events import EventLog  # noqa: E402
from server import examples, intake_runner, runner, sources, tasks, view  # noqa: E402

load_env()
app = FastAPI(title="DataMiner 工作台")


def _src(tid: str) -> dict:
    if tid.startswith("ex-"):
        if tid[3:] not in examples.names():
            raise HTTPException(404, "任务不存在")
        return examples.load(tid[3:])
    if not tasks.exists(tid):
        raise HTTPException(404, "任务不存在")
    return sources.live(tid)


def _need(tid: str, *statuses: str, idle: bool = True) -> dict:
    """操作前检查状态：示例只读；状态不对或接入正在起草时返回 409。"""
    if tid.startswith("ex-"):
        raise HTTPException(409, "示例任务只读")
    if not tasks.exists(tid):
        raise HTTPException(404, "任务不存在")
    meta = runner.refresh(tasks.load(tid))
    if meta["status"] not in statuses or (idle and meta.get("busy")):
        raise HTTPException(409, f"当前状态 {meta['status']} 不能执行这个操作")
    return meta


@app.get("/api/tasks")
def list_tasks():
    live = [{"id": m["id"], "title": m["title"], "status": m["status"], "created_at": m["created_at"], "example": False}
            for m in tasks.list_tasks()]
    return live + [{"id": f"ex-{n}", "title": f"示例 · {n}", "status": "done", "created_at": None, "example": True}
                   for n in examples.names()]


@app.post("/api/tasks", status_code=202)
def create(bg: BackgroundTasks, csv: UploadFile = File(...), description: str = Form(""),
           description_file: UploadFile | None = File(None), provider: str = Form("mock"), max_rounds: int = Form(5),
           full_trials: int = Form(5), low_trials: int | None = Form(None)):
    desc = "\n\n".join(x for x in (description.strip(), description_file.file.read().decode().strip() if description_file else "") if x)
    meta = tasks.create(csv.file.read(), csv.filename, desc, provider,
                        {"max_rounds": max_rounds, "full_trials": full_trials, "low_trials": low_trials})
    intake_runner.schedule(meta["id"])
    bg.add_task(intake_runner.start, meta["id"])
    return view.snapshot(sources.live(meta["id"]))


@app.get("/api/tasks/{tid}")
def get_task(tid: str):
    return view.snapshot(_src(tid))


class Answers(BaseModel):
    replies: dict[str, str]


@app.post("/api/tasks/{tid}/answers", status_code=202)
def answers(tid: str, body: Answers, bg: BackgroundTasks):
    _need(tid, "intake")
    intake_runner.schedule(tid)
    bg.add_task(intake_runner.answer, tid, body.replies)
    return {"ok": True}


@app.post("/api/tasks/{tid}/retry", status_code=202)
def retry(tid: str, bg: BackgroundTasks):
    _need(tid, "intake_failed")
    tasks.update(tid, status="intake", error=None)
    intake_runner.schedule(tid)
    bg.add_task(intake_runner.retry, tid)
    return {"ok": True}


@app.post("/api/tasks/{tid}/run")
def run(tid: str):
    _need(tid, "config_ready")
    runner.start(tid)
    return {"ok": True}


@app.post("/api/tasks/{tid}/stop")
def stop(tid: str):
    _need(tid, "running")
    runner.stop(tid)
    return {"ok": True}


@app.post("/api/tasks/{tid}/resume")
def resume(tid: str):
    _need(tid, "stopped", "failed")
    runner.start(tid)
    return {"ok": True}


class Accept(BaseModel):
    note: str = ""


@app.post("/api/tasks/{tid}/accept")
def accept(tid: str, body: Accept):
    _need(tid, "done")
    EventLog(str(tasks.events_db(tid)), tid).append("accepted", {"note": body.note})
    tasks.update(tid, status="accepted", note=body.note or None)
    return {"ok": True}


@app.post("/api/tasks/{tid}/clone")
def clone(tid: str):
    _need(tid, "config_ready", "running", "stopped", "failed", "done", "accepted", idle=False)
    return {"id": tasks.clone(tid)["id"]}


@app.get("/api/tasks/{tid}/events")
def events(tid: str):
    return _src(tid)["events"]


def _active(tid: str) -> bool:
    m = runner.refresh(tasks.load(tid))
    return m["status"] == "running" or m.get("busy", False)


@app.get("/api/tasks/{tid}/stream")
async def stream(tid: str, request: Request, after: int = 0):
    """推送 id 大于 Last-Event-ID（或 ?after=）的事件；任务空闲后发 idle 并结束，前端收到后关闭连接。"""
    if tid.startswith("ex-") or not tasks.exists(tid):
        raise HTTPException(404, "任务不存在")
    last = int(request.headers.get("last-event-id") or after)
    log = EventLog(str(tasks.events_db(tid)), tid)

    async def gen():
        nonlocal last
        yield "retry: 1000\n\n"
        while True:
            active = _active(tid)                       # 先看状态再读事件：结束前的最后一批事件不会漏
            for e in log.query():
                if e["id"] > last:
                    last = e["id"]
                    yield f"id: {e['id']}\nevent: task\ndata: {json.dumps(e, ensure_ascii=False, default=str)}\n\n"
            if not active:
                yield "event: idle\ndata: {}\n\n"
                return
            if await request.is_disconnected():
                return
            await asyncio.sleep(1)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/tasks/{tid}/artifacts/{name}")
def artifact(tid: str, name: str):
    files = _src(tid)["files"]
    if name not in files:
        raise HTTPException(404, "没有这个产出")
    return PlainTextResponse(files[name])


@app.get("/api/tasks/{tid}/bundle.zip")
def bundle(tid: str):
    src = _src(tid)
    if not (src["meta"].get("example") or src["meta"]["status"] == "accepted"):
        raise HTTPException(409, "确认报告后才能下载")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, t in src["files"].items():
            z.writestr(n, t)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{tid}.zip"'})


if (ROOT / "web" / "dist").exists():                     # 放在所有 /api 路由之后
    app.mount("/", StaticFiles(directory=ROOT / "web" / "dist", html=True), name="web")
