"""Ordinary conversation -> project structure (tasks, owners, deadlines, estimates, dependencies).

Nobody has to type a command or tag the bot. For each group message the agent:
  1. decides cheaply whether it could be about work at all (`looks_actionable`: "lol", "ok", greetings never reach a model),
  2. asks the LLM to propose structured items, given the team, the existing tasks and today's date,
  3. VALIDATES every proposal deterministically before anything is written: owners must be real team members, dependencies
     must point at real tasks and may not create a cycle, dates must be real and not in the past, duplicates of existing
     work are dropped, and low-confidence guesses are discarded (the agent stays quiet rather than inventing work).

The LLM understands language; this module decides what is allowed to change the project.
"""
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from sqlmodel import Session, select

from .config import settings
from .models import Member, Task
from .pipeline import llm
from .taskparse import parse_date

log = logging.getLogger("converse")
MAX_ITEMS = 4

# A cheap gate: only messages that could plausibly assign work, commit to it, or relate tasks reach the model.
_ACTIONABLE = re.compile(
    r"\b(need|needs|needed|should|must|have to|has to|will|going to|gonna|can you|could you|would you|please|pls|plz|let'?s|"
    r"make sure|work on|build|create|fix|send|share|finish|complete|implement|deploy|test|review|update|write|prepare|design|"
    r"set ?up|add|remove|change|check|handle|take care|own|assign|asap|urgent|deadline|due|by (?:today|tonight|tomorrow|monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday|eod|end of day|morning|evening|\d)|before|after|once|until|then|first|depends|"
    r"dependent|waiting for|blocked|prerequisite|kar do|karna|bhejo|banao|kal tak|"
    # meetings and times: "group meeting in 20 minutes", "stand-up at 10am", "demo on Monday"
    r"meeting|meet|call|stand-?up|sync|demo|session|presentation|class|lecture|workshop|interview|catch-?up|huddle|"
    r"in \d+ ?(?:min|mins|minutes|hour|hours|hrs?)|at \d{1,2}(?::\d{2})? ?(?:am|pm)?|\d{1,2}(?::\d{2})? ?(?:am|pm)|o'?clock|"
    r"tomorrow|tonight|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|'ll\b|@\w+", re.I)


def looks_actionable(text: str) -> bool:
    t = (text or "").strip()
    return len(t) >= 12 and len(t.split()) >= 3 and bool(_ACTIONABLE.search(t))


@dataclass
class Proposal:
    kind: str                                   # "task" | "dependency" | "meeting"
    confidence: float
    evidence: str = ""
    title: str = ""
    owner_id: int | None = None
    due: str | None = None
    est_minutes: int | None = None
    after: list[str] = field(default_factory=list)        # existing task ids, or "new:<title>" for a task in the same message
    task_id: str | None = None                            # dependency: this task ...
    dep_id: str | None = None                             # ... waits for this one
    starts_at: datetime | None = None                     # meeting: when it starts (timezone-aware UTC)
    attendee_ids: list[int] = field(default_factory=list)   # meeting: who must be there; empty = the whole team


SYSTEM = """You read ONE message from a team's project chat and report changes to the project plan, as JSON only:
{"items": [ ... ]}   (an empty list is the right answer for most messages)

Item kinds:
 {"kind":"task","title":"short imperative title, no names, no dates","owner":"team member name or null","due":"YYYY-MM-DD or null",
  "estimate_minutes":integer or null,"after":["existing task id such as T2, or the exact title of another task in this same output"],
  "confidence":0.0-1.0,"evidence":"the exact words that justify it"}
 {"kind":"dependency","task":"existing task id","after":"existing task id","confidence":0.0-1.0,"evidence":"exact words"}
 {"kind":"meeting","title":"short title such as Team meeting or Sprint demo","in_minutes":integer or null,"start":"YYYY-MM-DDTHH:MM in local time or null",
  "attendees":["team member names"] or [] for everyone,"official":true or false,"confidence":0.0-1.0,"evidence":"exact words"}

Meetings:
- A meeting is OFFICIAL (official=true) when it is a project, team, class, mentor, client, stand-up, review, demo or sync meeting or call that the
  team is expected to attend. Social plans (drinks, parties, games, food, hanging out, jokes dressed up as events) are NOT official: return nothing
  for them. A question about a meeting ("is there a meeting?") is not an announcement.
- Give EITHER in_minutes (for "in 20 minutes") OR start (a clock time or date resolved from NOW), never both. attendees: names only if the message
  says who must come; otherwise [] (everyone).
- A meeting or call between team members is official too, even between just two of them ("I want to meet Bo at 9pm", "call with Ada
  tomorrow morning"): attendees are the people named plus the speaker.
- A request attached to the meeting ("Ada please make sure everyone is there") is ALSO a task for that person.

Rules:
- A task is real PROJECT WORK that someone is asked or assigned to do, or promises to do ("Soham please fix the deployment", "I will build the
  login page", "we need the report by Friday"). Ignore greetings, jokes, small talk, personal errands (food, calls, lifts), opinions, questions
  to nobody, and reports about work that already exists (status updates are handled elsewhere). If unsure, return {"items": []}.
- owner: only a name from the TEAM list, and only if the message assigns it to that person (addressed by name or @name, or "I" meaning the
  speaker). "I will ..." means the speaker. Otherwise null. Never invent a person.
- A request addressed to a named person to hand something over ("Ada send me the file", "Bo I need the report by end of day") is a task
  FOR THAT PERSON ("Send the file", "Send the report", due that day). A @handle that contains a team member's name refers to that member.
- If something must wait for work that is NOT on the board yet ("the dashboard must wait for the design review that Ada does"), output a task
  item for that missing work (with its owner if one is named) AND a dependency item whose "after" is that task's exact title.
- due: only if the message states or clearly implies a date; resolve words like "tomorrow" or "Friday" from TODAY. Otherwise null.
- estimate_minutes: only if a duration is stated.
- after / dependency: ONLY when the message says one thing must happen first: "after", "once", "before", "depends on", "needs X first",
  "can't start until", "waiting for X to build Y". Use the id of an existing task whenever the thing already exists; never create a second
  task for work that is already on the board. A task cannot wait for itself.
- confidence: how sure you are this is real work / a real dependency (0.9+ for explicit assignments, 0.5 for guesses).

Examples (team: Ada, Bo):
 "lol that demo was crazy" -> {"items": []}
 "Group meeting in 20 minutes, Ada please make sure everyone is there" -> {"items":[{"kind":"meeting","title":"Team meeting","in_minutes":20,"start":null,"attendees":[],"official":true,"confidence":0.95,"evidence":"Group meeting in 20 minutes"},{"kind":"task","title":"Make sure everyone attends the team meeting","owner":"Ada","due":"<today>","estimate_minutes":null,"after":[],"confidence":0.85,"evidence":"Ada please make sure everyone is there"}]}
 "stand-up tomorrow at 10am, everyone join" -> {"items":[{"kind":"meeting","title":"Stand-up","in_minutes":null,"start":"<tomorrow>T10:00","attendees":[],"official":true,"confidence":0.9,"evidence":"stand-up tomorrow at 10am"}]}
 "Bo and Ada demo with the mentor on Monday at 4pm" -> {"items":[{"kind":"meeting","title":"Demo with the mentor","in_minutes":null,"start":"<Monday>T16:00","attendees":["Bo","Ada"],"official":true,"confidence":0.9,"evidence":"demo with the mentor on Monday at 4pm"}]}
 "I wanna meet Bo today 9 pm" (speaker Ada) -> {"items":[{"kind":"meeting","title":"Meeting with Bo","in_minutes":null,"start":"<today>T21:00","attendees":["Ada","Bo"],"official":true,"confidence":0.85,"evidence":"meet Bo today 9 pm"}]}
 "party sesh in 20 min anyone" -> {"items": []}
 "need a beer anyone up" -> {"items": []}
 "Ada send me the staging password" -> {"items":[{"kind":"task","title":"Send the staging password","owner":"Ada","due":null,"estimate_minutes":null,"after":[],"confidence":0.85,"evidence":"Ada send me the staging password"}]}
 "Bo I need that report by end of day" -> {"items":[{"kind":"task","title":"Send the report","owner":"Bo","due":"<today>","estimate_minutes":null,"after":[],"confidence":0.85,"evidence":"Bo I need that report by end of day"}]}
 "Bo can you grab me a coffee" -> {"items": []}
 "Ada please write the API docs by tomorrow" -> {"items":[{"kind":"task","title":"Write the API docs","owner":"Ada","due":"<tomorrow>","estimate_minutes":null,"after":[],"confidence":0.95,"evidence":"Ada please write the API docs by tomorrow"}]}
 "I will test the checkout once Bo finishes the payment API" (T3 = Build the payment API owned by Bo) -> {"items":[{"kind":"task","title":"Test the checkout","owner":"the speaker","due":null,"estimate_minutes":null,"after":["T3"],"confidence":0.9,"evidence":"I will test the checkout once Bo finishes the payment API"}]}
 "the dashboard must wait for the design review that Ada does" (T1 = Build the dashboard) -> {"items":[{"kind":"task","title":"Do the design review","owner":"Ada","due":null,"estimate_minutes":null,"after":[],"confidence":0.85,"evidence":"the dashboard must wait for the design review that Ada does"},{"kind":"dependency","task":"T1","after":"Do the design review","confidence":0.85,"evidence":"the dashboard must wait for the design review that Ada does"}]}
 "the dashboard can't start until the backend files are ready" (T1 = Build the dashboard, T2 = Send the backend files) -> {"items":[{"kind":"dependency","task":"T1","after":"T2","confidence":0.9,"evidence":"the dashboard can't start until the backend files are ready"}]}
"""


COMMAND = """
The speaker typed /task: they are explicitly asking you to put this on the plan, so return the task (or meeting) it describes even when it is
phrased as a request, in Hindi or Hinglish, or with typos, with a clean title and the right owner ("/task tejas can you get the report from
soham" is a task for Tejas titled "Get the report from Soham"). Two exceptions, each returned as the ONLY item:
 {"kind":"status","task":"existing task id","confidence":0.0-1.0,"evidence":"exact words"}   it reports on existing work instead
    ("pr is done", "login page ho gaya", "stuck on the API key")
 {"kind":"question","task":"existing task id or null","confidence":0.0-1.0,"evidence":"exact words"}   it asks about work instead
    ("pr hua kya", "is the login page done?")"""


def _context(speaker: Member, members: list[Member], tasks: list[Task], today: date, recent: list[str],
             now: datetime | None = None, finished: bool = False) -> str:
    owner = {m.id: m.name for m in members}
    clock = (now or datetime.now().astimezone()).astimezone()
    shown = [t for t in tasks if finished or t.status != "Complete"]
    board = "\n".join(f"{t.id}: {t.title} (owner {owner.get(t.owner_id, '?')}, {t.status})" for t in shown) or "(no open tasks)"
    return (f"TODAY: {today.isoformat()} ({today:%A})\nNOW: {clock:%Y-%m-%d %H:%M} (local time, {clock.tzname()})\nTEAM: {', '.join(m.name for m in members)}\nSPEAKER: {speaker.name}\n"
            f"{'TASKS' if finished else 'OPEN TASKS'}:\n{board}\n"
            + ("EARLIER MESSAGES (oldest first, for context only; act on MESSAGE alone):\n" + "\n".join(recent[-6:]) + "\n" if recent else ""))


async def ask(speaker: Member, text: str, members: list[Member], tasks: list[Task], today: date, recent: list[str] | None = None,
              now: datetime | None = None, command: bool = False) -> list[dict]:
    """The raw model answer (a list of dicts). [] if there is no LLM, it fails, or it says there is nothing.
    command: the speaker typed /task, so they want something on the plan (see COMMAND)."""
    if not settings.llm_enabled:
        return []
    ctx = _context(speaker, members, tasks, today, recent or [], now, finished=command)
    data = await llm.complete_json(SYSTEM + (COMMAND if command else ""), f"{ctx}\nMESSAGE: {text}", max_tokens=900)
    items = (data or {}).get("items") if isinstance(data, dict) else None
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def _person(name, members: list[Member], speaker: Member) -> int | None:
    n = str(name or "").strip().lower()
    if not n or n in ("null", "none", "unknown"):
        return None
    if n in ("the speaker", "speaker", "me", "i", "myself", speaker.name.lower()):
        return speaker.id
    exact = [m.id for m in members if n in (m.name.lower(), m.key.lower(), (m.tg_username or "").lower())]
    if exact:
        return exact[0]
    near = [m.id for m in members if m.name.lower().startswith(n) or n.startswith(m.name.lower())]
    return near[0] if len(near) == 1 else None


def _task_ref(ref, tasks: list[Task], batch_titles: set[str]) -> str | None:
    r = str(ref or "").strip()
    if re.fullmatch(r"[Tt]\d+", r):
        return r.upper() if any(t.id == r.upper() for t in tasks) else None
    if r.lower() in batch_titles:
        return "new:" + r.lower()
    hits = [t.id for t in tasks if t.title.lower() == r.lower()]
    return hits[0] if len(hits) == 1 else None


def _norm(title: str) -> set[str]:
    stop = {"the", "a", "an", "to", "for", "of", "and", "our", "my", "your", "on", "in", "with", "it", "me", "you"}
    return {w for w in re.findall(r"[a-z0-9]+", title.lower()) if w not in stop}


_GENERIC = {"build", "write", "create", "make", "do", "get", "send", "fix", "set", "setup", "up", "finish", "complete", "prepare", "work",
            "task", "project", "new", "update", "add", "check", "review"}


def grounded(text: str, task: Task) -> bool:
    """The message really talks about this task: at least one meaningful word of its title appears in the message. Stops the
    model from attaching a relationship to a task the message never mentioned."""
    words = _norm(task.title) - _GENERIC
    return bool(words & _norm(text)) if words else True


def find_duplicate(title: str, tasks: list[Task]) -> Task | None:
    """The task already on the board (open, or finished) that is the same work as `title`, if any."""
    mine = _norm(title)
    if not mine:
        return None
    for t in tasks:
        theirs = _norm(t.title)
        if theirs and len(mine & theirs) / min(len(mine), len(theirs)) >= 0.75 and len(mine & theirs) >= 2:
            return t
    return None


def is_duplicate(title: str, tasks: list[Task]) -> bool:
    """The same work is already on the board (open, or finished): do not create it twice."""
    return find_duplicate(title, tasks) is not None


def _meeting_start(raw: dict, now: datetime) -> datetime | None:
    """A real moment in the near future: `in_minutes` from now, or a clock time/date in local time. Never in the past."""
    mins = raw.get("in_minutes")
    if isinstance(mins, (int, float)) and not isinstance(mins, bool) and 1 <= mins <= 60 * 24 * 30:
        return (now + timedelta(minutes=float(mins))).astimezone(timezone.utc)
    if raw.get("start"):
        try:
            dt = datetime.fromisoformat(str(raw["start"]).strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        dt = dt.astimezone() if dt.tzinfo is None else dt                  # no zone given: the team's local time
        dt = dt.astimezone(timezone.utc)
        return dt if now - timedelta(minutes=1) <= dt <= now + timedelta(days=30) else None
    return None


def is_duplicate_meeting(title: str, start: datetime, existing: list) -> bool:
    """The same meeting announced twice (two people repeating it) is one meeting."""
    for a in existing:
        if getattr(a, "status", "scheduled") != "scheduled":
            continue
        close = abs((a.starts_at.astimezone(timezone.utc) - start).total_seconds()) <= 10 * 60
        if close and (_norm(a.title) & _norm(title) or close):
            return True
    return False


def validate(items: list[dict], speaker: Member, members: list[Member], tasks: list[Task], today: date,
             min_confidence: float | None = None, text: str | None = None, now: datetime | None = None,
             appointments: list | None = None) -> list[Proposal]:
    """Everything the model said, filtered through the rules in the module docstring."""
    floor = settings.auto_task_min_confidence if min_confidence is None else min_confidence
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    out: list[Proposal] = []
    batch: set[str] = {str(i.get("title", "")).strip().lower() for i in items if isinstance(i, dict) and i.get("kind") == "task"}
    for raw in items[:MAX_ITEMS]:
        if not isinstance(raw, dict):
            continue
        try:
            conf = float(raw.get("confidence", 0))
        except (TypeError, ValueError):
            continue
        if conf < floor:
            continue
        kind, evidence = raw.get("kind"), str(raw.get("evidence", ""))[:200]
        if kind == "meeting":
            if raw.get("official") is not True:                       # drinks, parties and jokes are not team meetings
                continue
            title = " ".join(str(raw.get("title", "")).split()).strip(" .,;:-")
            start = _meeting_start(raw, now)
            if not (3 <= len(title) <= 80) or start is None or is_duplicate_meeting(title, start, appointments or []):
                continue
            names = [str(n).strip().lower() for n in (raw.get("attendees") or []) if isinstance(n, str)]
            everyone = not names or any(n in ("everyone", "all", "team", "everybody") for n in names)
            who = [] if everyone else list(dict.fromkeys(i for i in (_person(n, members, speaker) for n in names) if i))
            out.append(Proposal("meeting", conf, evidence, title=title, starts_at=start, attendee_ids=who))
            continue
        if kind == "task":
            title = " ".join(str(raw.get("title", "")).split()).strip(" .,;:-")
            if not (3 <= len(title) <= 90) or is_duplicate(title, tasks):
                continue
            owner = _person(raw.get("owner"), members, speaker) or speaker.id     # unowned work belongs to whoever raised it
            due = None
            if raw.get("due"):
                d = parse_date(str(raw["due"]), today)
                due = d.isoformat() if d and d >= today else None
            est = raw.get("estimate_minutes")
            est = int(est) if isinstance(est, (int, float)) and 0 < est <= 60 * 24 * 14 else None
            after = []
            for ref in raw.get("after") or []:
                r = _task_ref(ref, tasks, batch - {title.lower()})
                if r and not r.startswith("new:") and text is not None and not grounded(text, next(t for t in tasks if t.id == r)):
                    continue                                  # the message never mentioned that task
                if r and r not in after:
                    after.append(r)
            out.append(Proposal("task", conf, evidence, title=title, owner_id=owner, due=due, est_minutes=est, after=after))
        elif kind == "dependency":
            a, b = _task_ref(raw.get("task"), tasks, batch), _task_ref(raw.get("after"), tasks, batch)
            by_id = {t.id: t for t in tasks}
            mentioned = all(x.startswith("new:") or text is None or grounded(text, by_id[x]) for x in (a, b) if x)
            if a and b and a != b and mentioned:         # either end may be a task created by this same message ("new:<title>")
                out.append(Proposal("dependency", conf, evidence, task_id=a, dep_id=b))
    return out


def apply(s: Session, speaker: Member, proposals: list[Proposal]) -> tuple[list[str], list[list[tuple[str, str]]]]:
    """Write the proposals. Returns (human lines for the confirmation, button rows to undo each change)."""
    from . import service
    members = {m.id: m for m in s.exec(select(Member)).all()}
    titles = {t.id: t.title for t in s.exec(select(Task)).all()}
    new_ids: dict[str, str] = {}
    lines: list[str] = []
    buttons: list[list[tuple[str, str]]] = []
    from html import escape
    for p in [x for x in proposals if x.kind == "task"]:
        deps = [new_ids.get(a[4:], None) if a.startswith("new:") else a for a in p.after]
        try:
            t = service.create_task(s, title=p.title, owner_id=p.owner_id, deadline=p.due, est_minutes=p.est_minutes,
                                    depends_on=[d for d in deps if d], requested_by_id=speaker.id)
        except ValueError as e:
            log.info("skipped %r: %s", p.title, e)
            continue
        new_ids[p.title.lower()] = t.id
        titles[t.id] = t.title
        bits = [f"<b>{escape(members[p.owner_id].name)}</b>" if p.owner_id == speaker.id else "{M%d}" % p.owner_id]    # a real @mention notifies them
        if p.due:
            bits.append(f"due {date.fromisoformat(p.due):%a %d %b}")
        if p.est_minutes:
            bits.append(f"~{p.est_minutes // 60}h{p.est_minutes % 60:02d}" if p.est_minutes >= 60 else f"~{p.est_minutes} min")
        if t.depends_on:
            bits.append("after " + ", ".join(escape(titles.get(d, d)) for d in t.depends_on.split(",")))
        lines.append(f"🆕 <b>{t.id} · {escape(t.title)}</b> → " + ", ".join(bits))
        buttons.append([(f"↩️ Undo {t.id}", f"dt:{t.id}")])
    for p in [x for x in proposals if x.kind == "meeting"]:
        appt = service.create_appointment(s, title=p.title, starts_at=p.starts_at, created_by_id=speaker.id,
                                          attendee_ids=p.attendee_ids, source_text=p.evidence)
        who = ", ".join("{M%d}" % i for i in p.attendee_ids) if p.attendee_ids else "everyone"
        lines.append(f"📅 <b>{escape(appt.title)}</b> {service.when(appt.starts_at)} · {who}. I'll remind them {appt.remind_min} minutes before.")
        buttons.append([("↩️ Cancel meeting", f"mc:{appt.id}")])
    for p in [x for x in proposals if x.kind == "dependency"]:
        a = new_ids.get(p.task_id[4:]) if p.task_id.startswith("new:") else p.task_id
        b = new_ids.get(p.dep_id[4:]) if p.dep_id.startswith("new:") else p.dep_id
        if not a or not b:
            continue
        try:
            service.add_dependency(s, a, b)
        except ValueError as e:
            log.info("skipped dependency %s -> %s: %s", a, b, e)
            continue
        lines.append(f"🔗 <b>{escape(titles[a])}</b> now waits for <b>{escape(titles[b])}</b>")
        buttons.append([(f"↩️ Unlink {a}", f"ul:{a}:{b}")])
    return lines, buttons


async def handle(s: Session, speaker: Member, text: str, today: date | None = None, recent: list[str] | None = None
                 ) -> tuple[list[str], list[list[tuple[str, str]]]]:
    """Message in -> (confirmation lines, undo buttons); both empty when the message is not about the plan."""
    today = today or date.today()
    members = list(s.exec(select(Member)).all())
    tasks = list(s.exec(select(Task)).all())
    from .models import Appointment
    now = datetime.now(timezone.utc)
    upcoming = list(s.exec(select(Appointment).where(Appointment.status == "scheduled")).all())
    proposals = validate(await ask(speaker, text, members, tasks, today, recent, now), speaker, members, tasks, today, text=text, now=now,
                         appointments=upcoming)
    if not proposals:
        return [], []
    return apply(s, speaker, proposals)


COMMAND_MIN_CONFIDENCE = 0.4          # /task is an explicit request: a lower bar than overhearing a conversation


@dataclass
class Command:
    """What a /task in plain words turned out to be."""
    kind: str                                   # "plan" (tasks/meetings written) | "status" | "question" | "duplicate" | "nothing"
    lines: list[str] = field(default_factory=list)
    buttons: list[list[tuple[str, str]]] = field(default_factory=list)
    task: Task | None = None                    # status / question / duplicate: the existing task it is about


async def handle_command(s: Session, speaker: Member, text: str, today: date | None = None, recent: list[str] | None = None) -> Command:
    """/task <plain words>: the model reads it like any message, knowing the speaker wants it on the plan."""
    today = today or date.today()
    members = list(s.exec(select(Member)).all())
    tasks = list(s.exec(select(Task)).all())
    from .models import Appointment
    now = datetime.now(timezone.utc)
    items = await ask(speaker, text, members, tasks, today, recent, now, command=True)
    by_id = {t.id: t for t in tasks}
    for raw in items:
        if raw.get("kind") in ("status", "question"):
            return Command(raw["kind"], task=by_id.get(str(raw.get("task") or "").strip().upper()))
    upcoming = list(s.exec(select(Appointment).where(Appointment.status == "scheduled")).all())
    proposals = validate(items, speaker, members, tasks, today, min_confidence=COMMAND_MIN_CONFIDENCE, text=text, now=now,
                         appointments=upcoming)
    if proposals:
        lines, buttons = apply(s, speaker, proposals)
        if lines:
            return Command("plan", lines, buttons)
    dup = next((d for i in items if i.get("kind") == "task" and (d := find_duplicate(str(i.get("title", "")), tasks))), None)
    return Command("duplicate", task=dup) if dup else Command("nothing")


SYSTEM_DEPS = """You are given the open tasks of a team project. Propose the dependencies that are OBVIOUS from their titles: pairs where one task
cannot sensibly start or finish before another is done (a dashboard waits for the backend it shows; deploying waits for what is deployed;
tests wait for the code under test; the demo video waits for the thing being demoed; anything waits for the files, credentials or access it
needs). Only strong relationships. Never a cycle, never a task waiting for itself, never a relationship that is merely "related".
Answer as JSON only: {"items":[{"kind":"dependency","task":"T3","after":"T1","confidence":0.0-1.0,"evidence":"why T3 needs T1"}]}
Use only the task ids given. An empty list is fine."""

INFER_MIN_CONFIDENCE = 0.75


async def infer_dependencies(s: Session) -> list[str]:
    """Link the obvious prerequisites between tasks that are already on the board. Returns what it linked, as plain lines."""
    from . import service
    if not settings.llm_enabled:
        raise ValueError("An LLM is needed to infer dependencies: set LLM_API_KEY and LLM_MODEL")
    members = {m.id: m for m in s.exec(select(Member)).all()}
    tasks = [t for t in s.exec(select(Task)).all() if t.status != "Complete"]
    if len(tasks) < 2:
        return []
    board = "\n".join(f"{t.id}: {t.title} (owner {members[t.owner_id].name}; already waits for: {t.depends_on or 'nothing'})" for t in tasks)
    data = await llm.complete_json(SYSTEM_DEPS, "OPEN TASKS:\n" + board, max_tokens=900)
    ids = {t.id: t for t in tasks}
    linked: list[str] = []
    for raw in ((data or {}).get("items") or [])[:12]:
        try:
            a, b, conf = str(raw.get("task")), str(raw.get("after")), float(raw.get("confidence", 0))
        except (TypeError, ValueError):
            continue
        if conf < INFER_MIN_CONFIDENCE or a not in ids or b not in ids:
            continue
        try:
            service.add_dependency(s, a, b)
        except ValueError:
            continue                                   # already there, a loop, or the same task: skip silently
        linked.append(f"{ids[a].title} now waits for {ids[b].title}")
    return linked
