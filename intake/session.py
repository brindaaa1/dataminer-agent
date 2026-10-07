"""起草 → 代码校验（失败回灌重写）→ 追问 → 回答 → …… → 写出 YAML。每一步都记成事件，录制和回放都用它。"""
import time

from pydantic import ValidationError

from intake.draft import draft, parse
from intake.profile import profile
from intake.schemas import REQUIRED, IntakeDraft, Question
from intake.validate import missing_confirmations, validate, value
from intake.writer import register, write

STATE_KEYS = ("csv", "description", "dataset_id", "out_dir", "max_rounds", "max_retries",
              "answers", "events", "round", "written", "profile", "confirmed", "shown", "last_draft", "prefs")
ACCEPT = ("确认", "采纳")          # 表单里原样提交推荐答案，或回答"确认"，都算采纳


class IntakeFailed(RuntimeError):
    pass


class IntakeSession:
    def __init__(self, llm, csv_path, description, dataset_id, out_dir, max_rounds=3, max_retries=2, prefs=None):
        """prefs：用户偏好（memory/preferences.recall），作为推荐答案放进起草上下文。"""
        self.llm, self.csv, self.description, self.dataset_id, self.out_dir = llm, csv_path, description, dataset_id, out_dir
        self.prefs = prefs
        self.max_rounds, self.max_retries = max_rounds, max_retries
        self.answers, self.events, self.round, self.written = {}, [], 0, None
        self.confirmed, self.shown, self.last_draft = {}, {}, None     # confirmed：必答项 → 用户确认时草稿里的值
        self.profile = profile(csv_path)
        self._log("profile", self.profile)

    def _log(self, type_, payload):
        self.events.append({"type": type_, "ts": time.time(), "round": self.round, "payload": payload})

    def step(self) -> list[Question]:
        """起草一轮，返回待回答的问题；没有问题时写出 YAML（self.written）。
        必答项由代码补齐：没确认过、或草稿改了已确认的值，都要（再）确认；写出时必答项一律用确认过的值。
        追问最多 max_rounds 轮；之后非必答问题不再追问，采用草稿中的值。"""
        self.round += 1
        feedback = None
        for _ in range(self.max_retries + 1):
            reply, prompt = draft(self.llm, self.profile, self.description, self.answers, feedback, self.prefs)
            self._log("llm_call", {"prompt": prompt, "response": reply})
            try:
                d = parse(reply)
            except (ValidationError, ValueError) as e:              # 格式不对也回灌，与决策节点的做法一致
                feedback = [f"输出不符合格式要求：{str(e)[:400]}"]
            else:
                feedback = validate(d, self.csv)
            if not feedback:
                break
            self._log("validation_failed", {"errors": feedback})
        else:
            raise IntakeFailed(feedback)
        self._log("draft", d.model_dump(mode="json"))
        self.last_draft = d.model_dump(mode="json")
        required = missing_confirmations(d, self.confirmed)
        extra = [q for q in d.questions if q.key not in self.answers and q.key not in {r.key for r in required}]
        qs = required + (extra if self.round <= self.max_rounds else [])
        self.shown = {q.key: q.recommended for q in qs}
        self._log("questions", [q.model_dump() for q in qs])
        if not qs:
            final = IntakeDraft.model_validate({**self.last_draft, **{k: self.confirmed[k]["raw"] for k in REQUIRED}})
            self.written = {**write(final, self.dataset_id, self.csv, self.out_dir), "draft": final.model_dump(mode="json")}
            self._log("written", {k: v for k, v in self.written.items() if k != "csv"})
        return qs

    def answer(self, replies: dict):
        """必答项：原样采纳推荐值（或回答"确认"）才记为已确认，记下的是当时草稿里的值；改了的交给 LLM 下一版草稿，再确认一次。"""
        d = IntakeDraft.model_validate(self.last_draft)
        replies = dict(replies)
        for k, v in replies.items():
            if k in REQUIRED:
                if v.strip() in ACCEPT or v == self.shown.get(k):
                    self.confirmed[k] = {"value": value(d, k), "raw": self.last_draft[k]}
                    replies[k] = f"已确认：{self.shown[k]}"      # 带上值：只存"确认"会冲掉之前的修改意见
                else:
                    self.confirmed.pop(k, None)
        self.answers.update(replies)
        self._log("answers", replies)

    def register(self, cfg: dict) -> dict:
        return register(cfg, self.written)

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in STATE_KEYS}

    @classmethod
    def from_json(cls, state: dict, llm):
        s = cls.__new__(cls)
        s.__dict__.update({k: state[k] for k in STATE_KEYS})
        s.llm = llm
        return s
