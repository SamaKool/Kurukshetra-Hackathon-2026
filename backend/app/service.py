"""Coordinator logic, independent of Telegram.

Every public function takes a DB session and returns an *outbox*: a list of messages to
send. transport.py delivers them through Telegram (or records them, in dev mode). This
keeps the brain testable without a bot token.
"""
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from html import escape

from sqlmodel import Session, select

from . import bus, converse
from .config import settings
from .models import AgentAction, Appointment, Correction, Event, Member, Message, Review, Task, now
from .pipeline.decide import gate, interpret
from .pipeline.rules import BLOCK, DONE, PROGRESS, heuristic
from .pipeline.types import Decision, Interpretation, TaskContext
from .taskparse import HELP as TASK_HELP
from .taskparse import is_task_command, parse_task
from .telemetry import decision_latency, replies_classified, reviews_created, span
from .textutil import looks_like_question, states_a_condition

STATUS_EMOJI = {"Pending": "⏳", "In Progress": "🔧", "Blocked": "⛔", "Complete": "✅"}
UPDATE_BUTTONS = lambda tid: [[("✅ Done", f"st:{tid}:done"), ("🔧 Working", f"st:{tid}:working"), ("⛔ Stuck", f"st:{tid}:stuck")]]
AFFIRM = re.compile(r"\b(sent|shared|done|given|gave|yes|ok|okay|sorted|fixed|bhej diya|de diya|ho gaya)\b|✅", re.I)
STOP = {"the", "a", "an", "is", "it", "to", "of", "and", "my", "i", "on", "for", "with", "s"}


def replan(kind: str, task_id: str | None = None) -> None:
    """Tell the scheduling engine that something it plans around changed. Never allowed to break a reply."""
    try:
        from .intel import engine
        engine.request(kind, task_id)
    except Exception:
        pass


def out(member_id, text, kind="system", task_id=None, buttons=None, reply_to=None):
    return {"member_id": member_id, "text": text, "kind": kind, "task_id": task_id,
            "buttons": buttons or [], "reply_to": reply_to}


# ---------------------------------------------------------------- helpers

def members_by_id(s: Session) -> dict[int, Member]:
    return {m.id: m for m in s.exec(select(Member)).all()}


def lead(s: Session) -> Member | None:
    return s.exec(select(Member).where(Member.role == "lead")).first()


def deps(task: Task) -> list[str]:
    return [d.strip() for d in task.depends_on.split(",") if d.strip()]


def dependents(s: Session, task_id: str) -> list[Task]:
    return [t for t in s.exec(select(Task)).all() if task_id in deps(t)]


def at_risk(s: Session, task: Task, tasks: dict[str, Task] | None = None) -> str | None:
    if task.status == "Complete":
        return None
    tasks = tasks or {t.id: t for t in s.exec(select(Task)).all()}
    for d in deps(task):
        dt = tasks.get(d)
        if dt and dt.status == "Blocked":
            return f"depends on '{dt.title}', which is Blocked"
        if dt and dt.status != "Complete" and dt.deadline and dt.deadline < date.today().isoformat():
            return f"depends on '{dt.title}', which is overdue"
    return None


def context(s: Session, task: Task) -> TaskContext:
    mb = members_by_id(s)
    recent = s.exec(select(Message).where(Message.task_id == task.id).order_by(Message.at.desc()).limit(3)).all()
    titles = {t.id: t.title for t in s.exec(select(Task)).all()}
    return TaskContext(
        task_id=task.id, title=task.title, status=task.status, owner=mb[task.owner_id].name,
        depends_on=[titles.get(d, d) for d in deps(task)],
        recent=[m.text for m in reversed(recent)], team=[m.name for m in mb.values()],
    )


def corrections(s: Session) -> list[tuple[str, str]]:
    rows = s.exec(select(Correction).order_by(Correction.at.desc()).limit(5)).all()
    return [(c.text, {"Complete": "completed", "In Progress": "progress", "Blocked": "blocked"}.get(c.right_status, "progress"))
            for c in reversed(rows)]


def match_task(s: Session, text: str, candidates: list[Task]) -> Task | None:
    """Fuzzy link a message to a task by aliases and title words."""
    low = text.lower()
    best, best_score = None, 0
    for t in candidates:
        score = 0
        for alias in [a.strip().lower() for a in t.aliases.split("|") if a.strip()]:
            if re.search(rf"\b{re.escape(alias)}\b", low):
                score = max(score, 2 + len(alias.split()))
        words = {w for w in re.findall(r"[a-z]+", t.title.lower()) if w not in STOP}
        score += len(words & set(re.findall(r"[a-z]+", low)))
        if score > best_score:
            best, best_score = t, score
    return best if best_score >= 2 else None


def snapshot(task: Task) -> str:
    return json.dumps({k: getattr(task, k) for k in (
        "status", "progress_pct", "eta", "blocker_text", "blocker_type", "blocked_by_id",
        "blocker_resolved", "proof_url")})


# ---------------------------------------------------------------- creating and removing tasks

def next_task_id(s: Session) -> str:
    nums = [int(t.id[1:]) for t in s.exec(select(Task)).all() if re.fullmatch(r"T\d+", t.id)]
    return f"T{max(nums, default=0) + 1}"


def create_task(s: Session, *, title: str, owner_id: int, deadline: str | None = None, est_minutes: int | None = None,
                depends_on: list[str] | tuple = (), requested_by_id: int | None = None) -> Task:
    """The one way a task comes into being (dashboard, Telegram, Slack). Raises ValueError on bad input."""
    title = " ".join((title or "").split())
    if len(title) < 3:
        raise ValueError("a task needs a title")
    if not s.get(Member, owner_id):
        raise ValueError("unknown owner")
    deps = list(dict.fromkeys(d.strip() for d in depends_on if d.strip()))
    missing = [d for d in deps if not s.get(Task, d)]
    if missing:
        raise ValueError(f"unknown prerequisite {', '.join(missing)}")
    if deadline:
        try:
            date.fromisoformat(deadline)
        except ValueError:
            raise ValueError("the deadline must look like 2026-10-25") from None
    if est_minutes is not None and est_minutes <= 0:
        raise ValueError("the estimate must be positive")
    t = Task(id=next_task_id(s), title=title, owner_id=owner_id, status="Pending", deadline=deadline or None,
             depends_on=",".join(deps), aliases=title.lower(), est_minutes=est_minutes, requested_by_id=requested_by_id)
    s.add(t)
    s.commit()
    s.refresh(t)
    bus.publish("board")
    replan("TASK_CREATED", t.id)
    return t


def delete_task(s: Session, task_id: str) -> bool:
    t = s.get(Task, task_id)
    if not t:
        return False
    for other in s.exec(select(Task)).all():                       # nothing may keep waiting for a task that is gone
        if task_id in deps(other):
            other.depends_on = ",".join(d for d in deps(other) if d != task_id)
            s.add(other)
    for model in (Event, Message, Review, AgentAction):
        for row in s.exec(select(model).where(model.task_id == task_id)).all():
            s.delete(row)
    s.delete(t)
    s.commit()
    bus.publish("board")
    replan("TASK_DELETED", task_id)
    return True


def add_dependency(s: Session, task_id: str, dep_id: str) -> None:
    """`task_id` now waits for `dep_id`. Raises ValueError for unknown tasks, self-dependencies and cycles."""
    from .intel.graph import ProjectGraph
    t, d = s.get(Task, task_id), s.get(Task, dep_id)
    if not t or not d:
        raise ValueError("unknown task")
    if task_id == dep_id:
        raise ValueError("a task cannot wait for itself")
    if dep_id in deps(t):
        raise ValueError("already a prerequisite")
    if ProjectGraph.from_db(s).path_between(task_id, dep_id):         # dep already (indirectly) waits for task: that would be a loop
        raise ValueError(f"{d.title} already waits for {t.title}, so this would be a loop")
    t.depends_on = ",".join([*deps(t), dep_id])
    s.add(t)
    s.commit()
    bus.publish("board")
    replan("DEPENDENCY_CHANGED", task_id)


def remove_dependency(s: Session, task_id: str, dep_id: str) -> bool:
    t = s.get(Task, task_id)
    if not t or dep_id not in deps(t):
        return False
    t.depends_on = ",".join(d for d in deps(t) if d != dep_id)
    s.add(t)
    s.commit()
    bus.publish("board")
    replan("DEPENDENCY_CHANGED", task_id)
    return True


def when(starts_at: datetime, now_: datetime | None = None) -> str:
    """"Sat 04:50 (in 20 min)": local time plus how far away it is."""
    now_ = now_ or datetime.now(timezone.utc)
    local = starts_at.astimezone()
    mins = int((starts_at.astimezone(timezone.utc) - now_).total_seconds() // 60)
    away = (f"in {mins} min" if mins < 90 else f"in {mins // 60} h {mins % 60:02d} min" if mins < 48 * 60 else f"in {mins // 1440} days") if mins > 0 else "now"
    return f"{local:%a %H:%M} ({away})"


def create_appointment(s: Session, *, title: str, starts_at: datetime, created_by_id: int | None = None,
                       attendee_ids: list[int] | tuple = (), source_text: str = "") -> Appointment:
    appt = Appointment(title=title, starts_at=starts_at.astimezone(timezone.utc), created_by_id=created_by_id,
                       attendee_ids=",".join(str(i) for i in attendee_ids), source_text=source_text[:300])
    s.add(appt)
    s.commit()
    s.refresh(appt)
    bus.publish("board")
    return appt


def cancel_appointment(s: Session, appointment_id: int) -> bool:
    a = s.get(Appointment, appointment_id)
    if not a or a.status != "scheduled":
        return False
    a.status = "cancelled"
    s.add(a)
    s.commit()
    bus.publish("board")
    return True


def handle_task_command(s: Session, member: Member, text: str, today: date | None = None) -> tuple[str, list[dict]]:
    """"task: Build login @soham due friday after T1" -> (reply for the chat it came from, notifications for others)."""
    people = list(s.exec(select(Member)).all())
    tasks = list(s.exec(select(Task)).all())
    p = parse_task(text, [(m.key, m.name) for m in people], [(t.id, t.title, t.aliases) for t in tasks], today or date.today())
    if p.errors:
        return ("⚠️ I did not create that task:\n" + "\n".join(f"• {escape(e)}" for e in p.errors) + "\n\n" + escape(TASK_HELP)), []
    owner = next((m for m in people if m.key == p.owner_key), member)
    try:
        t = create_task(s, title=p.title, owner_id=owner.id, deadline=p.due, est_minutes=p.est_minutes, depends_on=p.depends_on,
                        requested_by_id=member.id)
    except ValueError as e:
        return f"⚠️ I did not create that task: {escape(str(e))}", []
    titles = {x.id: x.title for x in tasks}
    bits = [f"owner <b>{escape(owner.name)}</b>"]
    if p.due:
        bits.append(f"due {date.fromisoformat(p.due):%a %d %b}")
    if p.est_minutes:
        bits.append(f"about {p.est_minutes // 60}h {p.est_minutes % 60:02d}m" if p.est_minutes >= 60 else f"about {p.est_minutes} min")
    if p.depends_on:
        bits.append("after " + ", ".join(escape(titles.get(d, d)) for d in p.depends_on))
    reply = f"✅ Created <b>{t.id} · {escape(t.title)}</b> ({', '.join(bits)})."
    notes = [] if owner.id == member.id else [out(owner.id, f"📌 {escape(member.name)} gave you a new task: <b>{escape(t.title)}</b>"
                                                    + (f", due {date.fromisoformat(p.due):%a %d %b}" if p.due else "") + ".",
                                                    "system", t.id, UPDATE_BUTTONS(t.id))]
    return reply, notes


# the strict form: an @owner, a due/est keyword or a prerequisite id. Written like that, /task is parsed exactly as typed.
_STRICT = re.compile(r"@\w|\b(?:due|est)\s+\S|\b(?:after|depends\s+on)\s+T\d+\b", re.I)


def recent_group(s: Session, limit: int = 6) -> list[str]:
    """The last group messages as "Name: text", oldest first, so the model knows who said what."""
    rows = s.exec(select(Message).where(Message.kind == "group").order_by(Message.id.desc()).limit(limit)).all()
    names = {m.id: m.name for m in s.exec(select(Member)).all()}
    return [f"{names.get(r.member_id, 'someone')}: {r.text}" for r in reversed(rows)]


async def task_command(s: Session, member: Member, text: str, channel: str = "telegram", ext_id: str | None = None,
                       today: date | None = None) -> list[dict]:
    """/task (or "task: ...") in any app. In the strict form it is parsed exactly; in plain words the LLM reads it like any message,
    so "/task tejas can you get the report from soham" is Tejas's task, "/task pr is done" is a status report and "/task pr hua kya"
    is a question. Without an LLM, or when the model finds nothing it can use, the strict parser answers (and explains itself)."""
    from .taskparse import PREFIX
    body = PREFIX.sub("", text or "", count=1).strip()
    if settings.llm_enabled and len(body) >= 3 and not _STRICT.search(body):
        c = await converse.handle_command(s, member, body, today, recent=recent_group(s))
        t = c.task
        if c.kind == "plan":
            return [out(member.id, "\n".join(c.lines), "confirm", None, c.buttons)]
        if c.kind == "status" and t:
            msg = record_in(s, member, body, t.id, channel=channel, ext_id=ext_id)
            if t.owner_id == member.id:
                return await decide_and_act(s, member, t, body, msg)
            owner = s.get(Member, t.owner_id)
            return await check_conflict(s, member, t, body, msg) or [
                out(member.id, f"<b>{t.id} · {escape(t.title)}</b> belongs to {escape(owner.name)}, so only they can update it. "
                               f"It is <b>{t.status}</b> right now.", "system", t.id)]
        if c.kind in ("status", "question"):
            if t is None:
                return [out(member.id, "I could not tell which task you mean. Name it or use its id, like T3.", "system")]
            owner = s.get(Member, t.owner_id)
            extra = f" ⛔ {escape(t.blocker_text)}" if t.status == "Blocked" and t.blocker_text else ""
            return [out(member.id, f"{STATUS_EMOJI.get(t.status, '')} <b>{t.id} · {escape(t.title)}</b> is <b>{t.status}</b> "
                                   f"(owner {escape(owner.name)}).{extra}", "system", t.id)]
        if c.kind == "duplicate" and t:
            return [out(member.id, f"That is already on the board: <b>{t.id} · {escape(t.title)}</b> "
                                   f"(owner {escape(s.get(Member, t.owner_id).name)}, {t.status}).", "system", t.id)]
    reply, notes = handle_task_command(s, member, text, today)
    return [out(member.id, reply, "confirm")] + notes


def may_undo(s: Session, member: Member, *, task: Task | None = None, appointment: Appointment | None = None) -> bool:
    """An undo button is visible to the whole group, so only the person who asked for the change, the owner or the lead may tap it."""
    if member.role == "lead":
        return True
    if task is not None:
        return member.id in (task.owner_id, task.requested_by_id) or task.requested_by_id is None
    if appointment is not None:
        return appointment.created_by_id in (None, member.id)
    return True


# ---------------------------------------------------------------- recording

def record_in(s, member, text, task_id=None, kind="reply", channel="telegram", ext_id=None, reply_to=None) -> Message:
    m = Message(member_id=member.id if member else None, task_id=task_id, direction="in", kind=kind,
                channel=channel, ext_id=ext_id, reply_to_ext=reply_to, text=text)
    s.add(m)
    s.commit()
    s.refresh(m)
    bus.publish("message", task_id=task_id)
    return m


def write_event(s, task, old, new, reason, engine, actor, interp: Interpretation | None = None,
                msg: Message | None = None, before: str | None = None) -> Event:
    ev = Event(task_id=task.id, old_status=old, new_status=new, reason=reason, engine=engine,
               confidence=interp.confidence if interp else None,
               evidence=msg.text if msg else None, message_id=msg.id if msg else None, actor=actor,
               trace=json.dumps(interp.trace if interp else []), before=before or snapshot(task))
    s.add(ev)
    s.commit()
    s.refresh(ev)
    bus.publish("event", task_id=task.id)
    return ev


# ---------------------------------------------------------------- applying a decision

def apply(s: Session, task: Task, decision: Decision, msg: Message | None, actor: str) -> list[dict]:
    i, new = decision.interp, decision.new_status
    mb = members_by_id(s)
    owner = mb[task.owner_id]
    before, old = snapshot(task), task.status
    outbox: list[dict] = []

    task.status = new
    task.last_update_at = now()
    task.reminder_count = 0
    if i.progress_pct is not None:
        task.progress_pct = i.progress_pct
    if i.eta:
        task.eta = i.eta

    if new == "Blocked":
        task.blocker_text, task.blocker_type, task.blocker_resolved = i.blocker_text, i.blocker_type, False
        task.blocked_by_id = None
        if i.blocked_by:
            hit = next((m for m in mb.values() if m.name.lower() == str(i.blocked_by).lower()), None)
            task.blocked_by_id = hit.id if hit else None
        if task.blocked_by_id is None:                      # fall back to the owner of an open dependency
            for d in deps(task):
                dt = s.get(Task, d)
                if dt and dt.status != "Complete" and dt.owner_id != task.owner_id:
                    task.blocked_by_id = dt.owner_id
                    task.blocker_type = task.blocker_type or "teammate"
                    break
    elif new == "Complete":
        task.progress_pct = 100
        task.proof_url = i.proof_url or task.proof_url
        task.blocker_text = task.blocker_type = None
        task.blocked_by_id = None
    elif new == "In Progress" and old == "Blocked":
        task.blocker_resolved = True

    s.add(task)
    s.commit()
    reason = {
        "Blocked": f"blocked: {task.blocker_text}",
        "Complete": "reported done" + (f" (proof: {task.proof_url})" if task.proof_url else ""),
        "In Progress": "progress update" + (f", {task.progress_pct}%" if task.progress_pct is not None else "")
                       + (f", ETA {task.eta}" if task.eta else ""),
    }[new]
    ev = write_event(s, task, old, new, reason, i.engine, actor, i, msg, before)

    # 1. confirm to the owner, with Undo
    line = f"{STATUS_EMOJI[new]} Got it: <b>{escape(task.title)}</b> → <b>{new}</b>"
    if new == "Blocked":
        line += f"\nBlocker recorded: <i>{escape(task.blocker_text or '')}</i>"
        if task.blocked_by_id:
            line += f"\nI've asked {escape(mb[task.blocked_by_id].name)} to help, and I'll check back with you."
    if new == "Complete" and not task.proof_url:
        line += "\nIf you have a link (PR, commit, file), reply with it and I'll attach it as proof."
    outbox.append(out(owner.id, line, "confirm", task.id, [[("↩️ Undo", f"undo:{ev.id}")]]))

    # 2. ask the unblocker
    if new == "Blocked" and task.blocked_by_id and task.blocked_by_id != owner.id:
        outbox.append(out(task.blocked_by_id,
            f"🙋 {escape(owner.name)} is blocked on <b>{escape(task.title)}</b>: <i>{escape(task.blocker_text or '')}</i>.\n"
            f"Can you help? Tap below or reply here once it's sorted.",
            "unblock_request", task.id, [[("✅ Sorted, tell them", f"unblock:{task.id}")]]))

    # 3. dependents: warn when blocked, release when done
    for dt in dependents(s, task.id):
        if dt.status == "Complete" or dt.owner_id == owner.id:
            continue
        if new == "Blocked":
            outbox.append(out(dt.owner_id,
                f"⚠️ Heads up: <b>{escape(task.title)}</b> is now Blocked ({escape(task.blocker_text or '')}). "
                f"Your task <b>{escape(dt.title)}</b> depends on it, so it's At Risk.", "system", dt.id))
        elif new == "Complete":
            outbox.append(out(dt.owner_id,
                f"🟢 <b>{escape(task.title)}</b> is Complete. Nothing stops <b>{escape(dt.title)}</b> now.", "system", dt.id))
    bus.publish("board")
    if old != new or i.eta or i.progress_pct is not None:
        replan({"Blocked": "TASK_BLOCKED", "Complete": "TASK_COMPLETED"}.get(new, "TASK_UPDATED"), task.id)
    return outbox


def send_to_review(s, task, decision: Decision, member: Member, msg: Message | None, source: str = "chat") -> list[dict]:
    i = decision.interp
    reviews_created.add(1, {"source": source})
    rv = Review(task_id=task.id, member_id=member.id, message_id=msg.id if msg else None, source=source,
                proposal=json.dumps({"status": decision.new_status, **i.to_dict()}), reasons=json.dumps(decision.reasons))
    s.add(rv)
    s.commit()
    s.refresh(rv)
    bus.publish("review")
    outbox = []
    if source == "chat":
        outbox.append(out(member.id, f"Thanks! I've asked the team lead to confirm the update on "
                                     f"<b>{escape(task.title)}</b>.", "confirm", task.id))
    ld = lead(s)
    if ld:
        q = escape((msg.text if msg else "")[:200])
        btns = [[("👍 Approve " + (decision.new_status or ""), f"rv:{rv.id}:approve")],
                [(f"{STATUS_EMOJI[st]} {st}", f"rv:{rv.id}:{st}") for st in ("In Progress", "Blocked", "Complete")]]
        where = "in the meeting" if source == "meeting" else ""
        outbox.append(out(ld.id,
            f"🤔 Please confirm {escape(member.name)}'s update {where} on <b>{escape(task.title)}</b>:\n“{q}”\n"
            f"My guess: <b>{decision.new_status}</b> ({i.engine}, confidence {i.confidence:.2f}).\n"
            f"Why I'm asking: {escape('; '.join(decision.reasons))}", "review", task.id, btns))
    return outbox


# ---------------------------------------------------------------- inbound entry points

async def handle_dm(s: Session, member: Member, text: str, *, reply_to: str | None = None,
                    ext_id: str | None = None, channel: str = "telegram", task_hint: str | None = None) -> list[dict]:
    if not member.awaiting_blocker_task and is_task_command(text):          # "task: ..." creates a task
        return await task_command(s, member, text, channel=channel, ext_id=ext_id)

    if not member.awaiting_blocker_task:                                    # an answer to one of the agent's own requests?
        from .intel import responses
        answered = await responses.try_handle(s, member, text, reply_to=reply_to, ext_id=ext_id, channel=channel)
        if answered is not None:
            return answered

    # A. answer to the "Stuck" button: this text *is* the blocker
    if member.awaiting_blocker_task:
        task = s.get(Task, member.awaiting_blocker_task)
        member.awaiting_blocker_task = None
        s.add(member)
        s.commit()
        msg = record_in(s, member, text, task.id, channel=channel, ext_id=ext_id, reply_to=reply_to)
        i = Interpretation(intent="blocked", confidence=0.95, blocker_text=text.strip()[:140], engine="button",
                           trace=["button: Stuck, then blocker described in text"])
        from .pipeline.rules import extract_fields
        extract_fields(text, i, context(s, task))
        i.blocker_text = text.strip()[:140]
        return apply(s, task, Decision("apply", "Blocked", i), msg, member.name)

    # B. link the message to a task
    task, linked_kind = None, None
    if task_hint:
        task = s.get(Task, task_hint)
    if task is None and reply_to:
        src = s.exec(select(Message).where(Message.ext_id == str(reply_to), Message.direction == "out",
                                          Message.channel == channel)).first()
        if src and src.task_id:
            task, linked_kind = s.get(Task, src.task_id), src.kind
    if task is None:
        # Most recent question we asked this person decides what a bare "sent them" refers to.
        last_out = s.exec(select(Message).where(Message.member_id == member.id, Message.direction == "out",
                          Message.kind.in_(["followup", "unblock_request"])).order_by(Message.at.desc())).first()
        owned_open = [t for t in s.exec(select(Task).where(Task.owner_id == member.id)).all() if t.status != "Complete"]
        task = match_task(s, text, owned_open)
        if task is None and last_out and last_out.kind == "unblock_request" and AFFIRM.search(text):
            task, linked_kind = s.get(Task, last_out.task_id), "unblock_request"
        if task is None and len(owned_open) == 1:
            task = owned_open[0]
        if task is None and owned_open:
            msg = record_in(s, member, text, None, channel=channel, ext_id=ext_id, reply_to=reply_to)
            member.pending_text_id = msg.id
            s.add(member)
            s.commit()
            return [out(member.id, "Which task is this about?", "clarify", None,
                        [[(t.title, f"pick:{t.id}")] for t in owned_open])]
        if task is None:
            record_in(s, member, text, None, channel=channel, ext_id=ext_id, reply_to=reply_to)
            return [out(member.id, "You have no open tasks right now 🎉", "system")]

    # C. the unblocker answering an unblock request
    if linked_kind == "unblock_request" and task.owner_id != member.id:
        msg = record_in(s, member, text, task.id, kind="unblock_reply", channel=channel, ext_id=ext_id, reply_to=reply_to)
        if AFFIRM.search(text):
            return resolve_blocker(s, task, member, text)
        owner = s.get(Member, task.owner_id)
        return [out(owner.id, f"💬 {escape(member.name)} replied about your blocker on <b>{escape(task.title)}</b>: “{escape(text)}”",
                    "system", task.id)]

    msg = record_in(s, member, text, task.id, channel=channel, ext_id=ext_id, reply_to=reply_to)
    if task.owner_id != member.id:
        return await check_conflict(s, member, task, text, msg)
    return await decide_and_act(s, member, task, text, msg)


async def decide_and_act(s, member, task, text, msg, quiet: bool = False) -> list[dict]:
    t0 = time.perf_counter()
    with span("coordinator.decide", task_id=task.id, member=member.key) as sp:
        interp = await interpret(text, context(s, task), corrections(s))
        decision = gate(interp, task.status)
        sp.set_attribute("engine", interp.engine)
        sp.set_attribute("intent", interp.intent)
        sp.set_attribute("confidence", interp.confidence)
        sp.set_attribute("action", decision.action)
    replies_classified.add(1, {"engine": interp.engine, "intent": interp.intent, "action": decision.action})
    decision_latency.record((time.perf_counter() - t0) * 1000, {"engine": interp.engine})
    if decision.action == "apply":
        return apply(s, task, decision, msg, member.name)
    if decision.action == "review":
        return send_to_review(s, task, decision, member, msg)
    if decision.action == "clarify":
        return [out(member.id, f"Where does <b>{escape(task.title)}</b> stand right now?", "followup", task.id,
                    UPDATE_BUTTONS(task.id))]
    if quiet:                                           # in a group, chit-chat is not answered
        return []
    return [out(member.id, "Noted 🙂 I only track task updates, so tell me whenever something changes.", "system", task.id)]


async def handle_group(s: Session, member: Member, text: str, ext_id=None, channel="telegram",
                       reply_to: str | None = None) -> list[dict]:
    """Group chat: catch teammates who believe something different from the board."""
    if is_task_command(text):                          # "task: ..." in the group creates a task
        return await task_command(s, member, text, channel=channel, ext_id=ext_id)
    from .intel import responses  # an answer to one of the agent's own messages?
    answered = await responses.try_handle(s, member, text, reply_to=reply_to, ext_id=ext_id, channel=channel)
    if answered is not None:
        return answered
    all_tasks = s.exec(select(Task)).all()
    claim = done_claim_about_others(s, member, text)
    if claim:
        t, clause = claim
        msg = record_in(s, member, text, t.id, kind="group", channel=channel, ext_id=ext_id)
        return await check_conflict(s, member, t, clause, msg)
    task = match_task(s, text, all_tasks)
    recent = recent_group(s)
    msg = record_in(s, member, text, task.id if task else None, kind="group", channel=channel, ext_id=ext_id)
    own_open = [t for t in all_tasks if t.owner_id == member.id and t.status != "Complete"]
    status_like = bool(DONE.search(text) or BLOCK.search(text) or PROGRESS.search(text))
    # 1. A status report about the speaker's own work. It does not have to name the task: "I've completed my task".
    #    Questions ("is the dashboard blocked on anything?") and talk about someone else are not the speaker's own status.
    others_named = any(m.id != member.id and re.search(rf"\b{re.escape(m.name)}\b", text, re.I) for m in s.exec(select(Member)).all())
    if status_like and not looks_like_question(text):
        target = task if (task and task.owner_id == member.id) else (
            own_open[0] if task is None and len(own_open) == 1 and not others_named else None)
        if target:
            return await decide_and_act(s, member, target, text, msg, quiet=True)
        if task is None and len(own_open) > 1 and DONE.search(text):
            member.pending_text_id = msg.id
            s.add(member)
            s.commit()
            return [out(member.id, "Which task is this about?", "clarify", None, [[(t.title, f"pick:{t.id}")] for t in own_open])]
    # 2. New work, an owner, a deadline or a dependency stated in ordinary words: no command and no tagging needed.
    if settings.auto_tasks and converse.looks_actionable(text):
        lines, buttons = await converse.handle(s, member, text, recent=recent)
        if lines:
            return [out(member.id, "\n".join(lines), "confirm", None, buttons)]
    # 3. Otherwise the older behaviour: a mention of someone else's task is checked against the board.
    if task is None or looks_like_question(text):               # a question about a task is not a report about it
        return []
    if task.owner_id == member.id:
        return await decide_and_act(s, member, task, text, msg, quiet=True)
    return await check_conflict(s, member, task, text, msg)


def done_claim_about_others(s: Session, member: Member, text: str) -> tuple[Task, str] | None:
    """Clause by clause, so "integration's done, starting checkout" checks the integration claim
    on its own instead of matching the speaker's own checkout task."""
    if states_a_condition(text) or looks_like_question(text):         # "needs X done first", "is X done?": not a claim
        return None
    others = [t for t in s.exec(select(Task)).all() if t.owner_id != member.id and t.status != "Complete"]
    for clause in re.split(r"[,;.!]| and | but | so ", text):
        t = match_task(s, clause, others)
        if t and DONE.search(clause):
            return t, clause.strip()
    return None


async def check_conflict(s, member, task, text, msg) -> list[dict]:
    if states_a_condition(text) or looks_like_question(text):          # "needs X done first" / "is X done?" claim nothing
        return []
    claims_done = bool(DONE.search(text)) or heuristic(text, context(s, task)).intent == "completed"
    if not claims_done or task.status == "Complete":
        return []
    owner = s.get(Member, task.owner_id)
    detail = f" ({escape(task.blocker_text)})" if task.status == "Blocked" and task.blocker_text else ""
    s.add(Message(member_id=member.id, task_id=task.id, direction="in", kind="conflict", channel="system",
                  text=f"{member.name} believes '{task.title}' is done; board says {task.status}"))
    s.commit()
    bus.publish("conflict", task_id=task.id)
    return [
        out(member.id, f"🔎 Quick check: you mentioned <b>{escape(task.title)}</b> as done, but {escape(owner.name)} "
                       f"last reported it <b>{task.status}</b>{detail}. Worth confirming before you build on it.",
            "conflict", task.id),
        out(owner.id, f"🔎 {escape(member.name)} thinks <b>{escape(task.title)}</b> is done, but it's marked "
                      f"<b>{task.status}</b>{detail}. Is the board out of date?", "followup", task.id, UPDATE_BUTTONS(task.id)),
    ]


def resolve_blocker(s, task: Task, by: Member, text: str) -> list[dict]:
    task.blocker_resolved = True
    s.add(task)
    s.commit()
    owner = s.get(Member, task.owner_id)
    s.add(Message(member_id=by.id, task_id=task.id, direction="in", kind="system", channel="system",
                  text=f"{by.name} resolved the blocker: {text}"))
    s.commit()
    bus.publish("board")
    replan("BLOCKER_RESOLVED", task.id)
    outbox = [
        out(by.id, f"Thanks {escape(by.name)}! I'll let {escape(owner.name)} know.", "system", task.id),
        out(owner.id, f"🔓 {escape(by.name)} says your blocker on <b>{escape(task.title)}</b> "
                      f"(<i>{escape(task.blocker_text or '')}</i>) is sorted: “{escape(text)}”.\nIs it done now?",
            "followup", task.id, UPDATE_BUTTONS(task.id)),
    ]
    # If the unblocker owns the dependency itself (e.g. the owner of "Get sandbox test credentials"), ask about it too.
    for d in deps(task):
        dt = s.get(Task, d)
        if dt and dt.owner_id == by.id and dt.status != "Complete":
            outbox.append(out(by.id, f"Does that mean <b>{escape(dt.title)}</b> is done too?", "followup", dt.id,
                              UPDATE_BUTTONS(dt.id)))
    return outbox


async def handle_pick(s, member: Member, task_id: str) -> list[dict]:
    msg = s.get(Message, member.pending_text_id) if member.pending_text_id else None
    member.pending_text_id = None
    s.add(member)
    s.commit()
    task = s.get(Task, task_id)
    if not msg or not task:
        return []
    msg.task_id = task.id
    s.add(msg)
    s.commit()
    return await decide_and_act(s, member, task, msg.text, msg)


def handle_button(s: Session, member: Member, task_id: str, choice: str) -> list[dict]:
    task = s.get(Task, task_id)
    if not task or task.owner_id != member.id:
        return [out(member.id, "Only the task owner can update that one.", "system")]
    if choice == "stuck":
        member.awaiting_blocker_task = task.id
        s.add(member)
        s.commit()
        return [out(member.id, f"What's blocking <b>{escape(task.title)}</b>? One line is enough "
                               f"(e.g. “waiting for test credentials from the platform team”).", "followup", task.id)]
    new = {"done": "Complete", "working": "In Progress"}[choice]
    i = Interpretation(intent={"Complete": "completed", "In Progress": "progress"}[new], confidence=1.0,
                       engine="button", trace=[f"button: {choice}"])
    msg = record_in(s, member, f"[button] {choice}", task.id)
    return apply(s, task, Decision("apply", new, i), msg, member.name)


def handle_unblock_button(s, member: Member, task_id: str) -> list[dict]:
    task = s.get(Task, task_id)
    return resolve_blocker(s, task, member, "marked as sorted") if task else []


# ---------------------------------------------------------------- lead actions

def undo(s: Session, event_id: int, actor: str) -> list[dict]:
    ev = s.get(Event, event_id)
    if not ev or ev.reverted:
        return []
    task = s.get(Task, ev.task_id)
    last = s.exec(select(Event).where(Event.task_id == task.id, Event.reverted == False)  # noqa: E712
                  .order_by(Event.at.desc())).first()
    if last.id != ev.id:
        return [out(s.get(Member, task.owner_id).id, "That change was already followed by a newer one, so I can't undo it.", "system")]
    before = json.loads(ev.before)
    cur = task.status
    for k, v in before.items():
        setattr(task, k, v)
    ev.reverted = True
    s.add_all([task, ev])
    s.commit()
    write_event(s, task, cur, task.status, f"undo of event #{ev.id}", "undo", actor)
    bus.publish("board")
    replan("TASK_UPDATED", task.id)
    return [out(task.owner_id, f"↩️ Undone. <b>{escape(task.title)}</b> is back to <b>{task.status}</b>.", "system", task.id)]


def resolve_review(s: Session, review_id: int, verdict: str, actor: str) -> list[dict]:
    rv = s.get(Review, review_id)
    if not rv or rv.status != "pending":
        return []
    p = json.loads(rv.proposal)
    status = p["status"] if verdict == "approve" else verdict
    if status not in STATUS_EMOJI:
        return []
    task = s.get(Task, rv.task_id)
    msg = s.get(Message, rv.message_id) if rv.message_id else None
    i = Interpretation(**{k: p[k] for k in Interpretation.__dataclass_fields__ if k in p})
    i.intent = {"Complete": "completed", "In Progress": "progress", "Blocked": "blocked", "Pending": "progress"}[status]
    if verdict != "approve":
        i.engine, i.confidence = "lead", 1.0
        if msg:
            s.add(Correction(text=msg.text, wrong_status=p["status"] or "", right_status=status))
    i.trace = i.trace + [f"lead {actor}: {'approved' if verdict == 'approve' else 'set ' + status}"]
    rv.status, rv.resolved_status = ("approved" if verdict == "approve" else "overridden"), status
    s.add(rv)
    s.commit()
    if status == "Pending":
        task.status = "Pending"
        s.add(task)
        s.commit()
        return []
    return apply(s, task, Decision("apply", status, i), msg, f"{actor} (lead)")


# ---------------------------------------------------------------- follow-ups

def followups(s: Session, force: bool = False, task_id: str | None = None) -> list[dict]:
    mb = members_by_id(s)
    cutoff = now() - timedelta(minutes=settings.stale_after_min)
    outbox = []
    tasks = [s.get(Task, task_id)] if task_id else s.exec(select(Task)).all()
    for t in tasks:
        if not t or t.status == "Complete":
            continue
        owner = mb[t.owner_id]
        stale = t.last_update_at < cutoff and (t.last_followup_at is None or t.last_followup_at < cutoff)
        if not (force or stale) or (not force and t.reminder_count >= 2):
            continue
        due = ""
        if t.deadline:
            try:
                due = f" It's due {date.fromisoformat(t.deadline):%a %d %b}."
            except ValueError:
                due = f" It's due {t.deadline}."
        last = ""
        if t.status == "Blocked" and t.blocker_text:
            last = f"\nLast time: <i>{escape(t.blocker_text)}</i>" + (" (marked sorted since)" if t.blocker_resolved else "") + ". Any change?"
        elif t.progress_pct is not None:
            last = f"\nLast update: {t.progress_pct}%" + (f", ETA {escape(t.eta)}" if t.eta else "") + "."
        outbox.append(out(owner.id, f"Hi {escape(owner.name)} 👋 How's <b>{escape(t.title)}</b> going?{due}{last}\n"
                                    f"Tap a button or just reply in your own words.", "followup", t.id, UPDATE_BUTTONS(t.id)))
        t.last_followup_at = now()
        t.reminder_count += 1
        s.add(t)
    s.commit()
    return outbox


# ---------------------------------------------------------------- read models for the dashboard

def board(s: Session) -> dict:
    mb = members_by_id(s)
    tasks = {t.id: t for t in s.exec(select(Task)).all()}
    rows = []
    for t in tasks.values():
        last = s.exec(select(Event).where(Event.task_id == t.id, Event.reverted == False)  # noqa: E712
                      .order_by(Event.at.desc())).first()
        rows.append({
            **t.model_dump(), "owner": mb[t.owner_id].name, "owner_key": mb[t.owner_id].key,
            "blocked_by": mb[t.blocked_by_id].name if t.blocked_by_id else None,
            "depends_on_titles": [tasks[d].title for d in deps(t) if d in tasks],
            "at_risk": at_risk(s, t, tasks),
            "last_event": last.model_dump() if last else None,
        })
    rows.sort(key=lambda r: r["id"])
    counts = {st: sum(1 for r in rows if r["status"] == st) for st in STATUS_EMOJI}
    return {
        "tasks": rows,
        "members": [{"id": m.id, "key": m.key, "name": m.name, "role": m.role, "connected": bool(m.linked_channels() or m.tg_user_id), "channels": m.linked_channels()}
                    for m in mb.values()],
        "counts": {**counts, "At Risk": sum(1 for r in rows if r["at_risk"])},
        "pending_reviews": len(s.exec(select(Review).where(Review.status == "pending")).all()),
    }


def timeline(s: Session, task_id: str) -> dict:
    evs = s.exec(select(Event).where(Event.task_id == task_id).order_by(Event.at)).all()
    msgs = s.exec(select(Message).where(Message.task_id == task_id).order_by(Message.at)).all()
    mb = members_by_id(s)
    items = [{"type": "event", "at": e.at, **e.model_dump(), "trace": json.loads(e.trace)} for e in evs]
    items += [{"type": "message", "at": m.at, **m.model_dump(), "who": mb[m.member_id].name if m.member_id else "bot"}
              for m in msgs]
    items.sort(key=lambda x: x["at"])
    return {"task": s.get(Task, task_id), "items": items}


def activity(s: Session, limit: int = 40) -> list[dict]:
    mb = members_by_id(s)
    titles = {t.id: t.title for t in s.exec(select(Task)).all()}
    evs = s.exec(select(Event).order_by(Event.at.desc()).limit(limit)).all()
    msgs = s.exec(select(Message).order_by(Message.at.desc()).limit(limit)).all()
    items = [{"type": "event", "at": e.at, "task": titles.get(e.task_id), "task_id": e.task_id,
              "text": f"{e.old_status} → {e.new_status}: {e.reason}", "engine": e.engine,
              "confidence": e.confidence, "actor": e.actor, "reverted": e.reverted} for e in evs]
    def names(text: str) -> str:                                       # {M3} mention tokens read as @Samarth on the dashboard
        return re.sub(r"\{M(\d+)\}", lambda x: f"@{mb[int(x.group(1))].name}" if int(x.group(1)) in mb else "@someone", text or "")
    items += [{"type": "message", "at": m.at, "task": titles.get(m.task_id), "task_id": m.task_id, "text": names(m.text),
               "direction": m.direction, "kind": m.kind, "who": mb[m.member_id].name if m.member_id else "bot"}
              for m in msgs]
    items.sort(key=lambda x: x["at"], reverse=True)
    return items[:limit]


def reviews(s: Session) -> list[dict]:
    mb = members_by_id(s)
    rows = s.exec(select(Review).where(Review.status == "pending").order_by(Review.at)).all()
    res = []
    for r in rows:
        msg = s.get(Message, r.message_id) if r.message_id else None
        res.append({**r.model_dump(), "proposal": json.loads(r.proposal), "reasons": json.loads(r.reasons),
                    "member": mb[r.member_id].name if r.member_id else None, "text": msg.text if msg else None,
                    "task": s.get(Task, r.task_id).title})
    return res
