"""Channel layer: one interface for Telegram, Slack and Microsoft Teams.

The coordinator (service.py) returns an outbox of platform-neutral messages written in a
tiny HTML subset (<b>, <i>) with optional buttons as (label, callback-data) pairs. This
module picks each member's channel and lets the adapter render it natively:
Telegram HTML + inline keyboard, Slack mrkdwn + Block Kit buttons, Teams markdown +
Adaptive Card Action.Submit. Button taps from every platform come back as the same
callback strings ("st:T2:done", "unblock:T2", "rv:7:approve") and go through
handle_callback, so business logic never knows which app was used.
"""
import html
import logging
import re
from datetime import datetime, timezone
from typing import Protocol

from sqlmodel import Session, select

from .. import bus, service
from ..config import settings
from ..models import Appointment, Member, Message, Setting, Task
from ..telemetry import messages_sent

log = logging.getLogger("channels")

# When each chat app last really talked to us (proof that its side is configured: URLs set, bot started).
STATS: dict[str, str | None] = {"slack_real": None, "telegram": None}


def stamp(key: str) -> None:
    STATS[key] = datetime.now(timezone.utc).isoformat()


class Channel(Protocol):
    name: str

    def ready(self) -> bool: ...

    def linked(self, m: Member) -> bool: ...

    async def send(self, m: Member, item: dict) -> str | None: ...


REGISTRY: dict[str, Channel] = {}


def register(ch: Channel) -> None:
    REGISTRY[ch.name] = ch


# ---------------------------------------------------------------- formatting

def to_plain(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def to_slack(text: str) -> str:
    """Telegram-style HTML -> Slack mrkdwn. Slack needs &, <, > escaped in plain text."""
    t = re.sub(r"<b>(.*?)</b>", r"*\1*", text, flags=re.S)
    t = re.sub(r"<i>(.*?)</i>", r"_\1_", t, flags=re.S)
    t = html.unescape(re.sub(r"<[^>]+>", "", t))
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_markdown(text: str) -> str:
    """Telegram-style HTML -> Markdown (Teams TextBlock, dashboard)."""
    t = re.sub(r"<b>(.*?)</b>", r"**\1**", text, flags=re.S)
    t = re.sub(r"<i>(.*?)</i>", r"_\1_", t, flags=re.S)
    return html.unescape(re.sub(r"<[^>]+>", "", t))


# ---------------------------------------------------------------- routing

def pick(m: Member) -> Channel | None:
    """Preferred channel if connected, else the first connected one."""
    order = ([m.preferred_channel] if m.preferred_channel else []) + ["slack", "teams", "telegram"]
    for name in order:
        ch = REGISTRY.get(name)
        if ch and ch.ready() and ch.linked(m):
            return ch
    return None


async def deliver(s: Session, outbox: list[dict], group_reply: dict | None = None) -> list[tuple[str, str | None]]:
    """Send each item to its recipient's channel. Returns [(channel, ext_id)] in order (for correlation).

    Two cases are answered *in the Telegram group* instead of a private chat:
      * `group_reply` = {"member_id", "chat_id", "message_id"}: the person just wrote in that group, so the answer goes
        there as a reply to their message (the natural place for it);
      * anyone who has identified themselves in the group but never opened a private chat with the bot: without this
        their messages would have nowhere to go and be silently dropped.
    """
    results: list[tuple[str, str | None]] = []
    people = {m.id: m for m in s.exec(select(Member)).all()} if outbox else {}
    tg = REGISTRY.get("telegram")
    can_group = tg is not None and tg.ready() and hasattr(tg, "send_group")
    for item in outbox:
        member = people.get(item["member_id"]) if item.get("member_id") else None
        if member is None:
            continue
        ext_id, channel = None, "dev"
        target = None                                           # (telegram chat id, message to reply to)
        raw_text = item["text"]
        ch = None
        if can_group and group_reply and member.id == group_reply["member_id"]:
            target = (group_reply["chat_id"], group_reply["message_id"])
        else:
            ch = pick(member)
            if ch is None and can_group and member.tg_user_id and (chat := group_chat_id(s)):
                target = (chat, None)
        if target is not None:
            try:
                body = expand_mentions(raw_text, people, "telegram")
                if not target[1]:                                   # not a reply: say who it is for
                    body = expand_mentions("{M%d} " % member.id, people, "telegram") + body
                ext_id = await tg.send_group(target[0], body, markup=item.get("buttons") or None, reply_to=target[1])
                channel = "telegram"
                messages_sent.add(1, {"channel": "telegram", "result": "ok"})
            except Exception as e:
                log.warning("group reply to %s failed: %s", member.name, e)
                channel = "failed"
                messages_sent.add(1, {"channel": "telegram", "result": "error"})
        elif ch is not None:
            try:
                ext_id = await ch.send(member, {**item, "text": expand_mentions(raw_text, people, "html")})
                channel = ch.name
                messages_sent.add(1, {"channel": ch.name, "result": "ok"})
            except Exception as e:
                log.warning("send via %s to %s failed: %s", ch.name, member.name, e)
                channel = "failed"
                messages_sent.add(1, {"channel": ch.name, "result": "error"})
        s.add(Message(member_id=member.id, task_id=item.get("task_id"), direction="out", kind=item["kind"],
                      channel=channel, ext_id=str(ext_id) if ext_id is not None else None,
                      text=to_plain(item["text"])))
        s.commit()
        results.append((channel, str(ext_id) if ext_id is not None else None))
    if outbox:
        bus.publish("message")
    return results


# ---------------------------------------------------------------- group chat (the agent's proactive messages)

_MENTION = re.compile(r"\{M(\d+)\}")


def expand_mentions(text: str, members: dict[int, Member], style: str) -> str:
    """{M<id>} -> a real mention that notifies the person (Telegram tg://user link, Slack <@U...>), else bold or plain @name."""
    def sub(m: re.Match) -> str:
        mem = members.get(int(m.group(1)))
        if mem is None:
            return "someone"
        if style == "telegram" and mem.tg_user_id:
            return f'<a href="tg://user?id={mem.tg_user_id}">{html.escape(mem.name)}</a>'
        if style == "plain":
            return f"@{mem.name}"
        if style == "slack":
            return f"<@{mem.slack_user_id}>" if mem.slack_user_id else f"*@{mem.name}*"
        return f"<b>@{html.escape(mem.name)}</b>"
    return _MENTION.sub(sub, text)


def group_chat_id(s: Session) -> int | None:
    raw = settings.telegram_group_chat_id
    if not raw:
        row = s.get(Setting, "tg_group_chat_id")
        raw = row.value if row else ""
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


_SLACK_CHANNEL = re.compile(r"^[CG][A-Z0-9]{8,}$")        # a public or private channel, never a DM


def _group_destinations(s: Session) -> list[tuple[str, object, object]]:
    """Where the agent speaks to the whole team: the Telegram group and the Slack project channel, whichever exist."""
    dests: list[tuple[str, object, object]] = []
    tg, chat = REGISTRY.get("telegram"), group_chat_id(s)
    if tg is not None and tg.ready() and chat and hasattr(tg, "send_group"):
        dests.append(("telegram", tg, chat))
    sl, ch = REGISTRY.get("slack"), settings.slack_project_channel
    if sl is not None and sl.ready() and ch and _SLACK_CHANNEL.match(ch) and hasattr(sl, "post"):
        dests.append(("slack", sl, ch))
    return dests


async def deliver_group(s: Session, text: str, task_id: str | None = None, kind: str = "agent_action",
                        recipient_ids: list[int] | None = None) -> tuple[str | None, str]:
    """Post to the team's group chats (Telegram group and/or Slack project channel), mentioning people.
    Returns (platform message ids joined by ';', channel such as "telegram+slack"). With no group chat configured the
    message goes to the *first* recipient (the person being asked) on their own channel, so the loop still works
    in development. channel == "failed": a send was attempted and broke (retry later); "unreachable": apps are
    connected but this person has not linked any of them (nobody else receives a message addressed to them)."""
    members = {m.id: m for m in s.exec(select(Member)).all()}
    dests = _group_destinations(s)
    if dests:
        ids: list[str] = []
        sent_via: list[str] = []
        for name, ch, target in dests:
            try:
                if name == "telegram":
                    ext = await ch.send_group(target, expand_mentions(text, members, "telegram"))
                else:                                           # Slack mrkdwn; mentions are added after escaping
                    ext = await ch.post(target, expand_mentions(to_slack(text), members, "slack"), None)
                sent_via.append(name)
                if ext is not None:
                    ids.append(str(ext))
            except Exception as e:
                log.warning("group send via %s failed: %s", name, e)
        channel, ext_id = ("+".join(sent_via) if sent_via else "failed"), (";".join(ids) or None)
        s.add(Message(member_id=None, task_id=task_id, direction="out", kind=kind, channel=channel, ext_id=ext_id,
                      text=to_plain(expand_mentions(text, members, "plain"))))
        s.commit()
        bus.publish("message", task_id=task_id)
        return ext_id, channel
    ids_ = [i for i in (recipient_ids or [int(x) for x in _MENTION.findall(text)]) if i in members][:1]
    if ids_ and pick(members[ids_[0]]) is None and any(ch.ready() for ch in REGISTRY.values()):
        s.add(Message(member_id=ids_[0], task_id=task_id, direction="out", kind=kind, channel="unreachable",
                      text=to_plain(expand_mentions(text, members, "plain"))))
        s.commit()
        bus.publish("message", task_id=task_id)
        return None, "unreachable"
    outbox = [{"member_id": i, "text": expand_mentions(text, members, "html"), "kind": kind, "task_id": task_id,
               "buttons": [], "reply_to": None} for i in ids_]
    results = await deliver(s, outbox)
    return (results[0][1], results[0][0]) if results else (None, "failed")


async def handle_callback(s: Session, member: Member, data: str) -> list[dict]:
    """Button taps from any platform (and the dashboard's dev panel)."""
    parts = data.split(":")
    kind = parts[0]
    if kind == "st":
        return service.handle_button(s, member, parts[1], parts[2])
    if kind == "unblock":
        return service.handle_unblock_button(s, member, parts[1])
    if kind == "undo":
        return service.undo(s, int(parts[1]), member.name)
    if kind == "rv":
        if member.role != "lead":
            return [service.out(member.id, "Only the team lead can resolve reviews.")]
        return service.resolve_review(s, int(parts[1]), parts[2], member.name)
    if kind == "pick":
        return await service.handle_pick(s, member, parts[1])
    if kind == "dt":                                      # undo a task the agent created from conversation
        t = s.get(Task, parts[1])
        title = t.title if t else parts[1]
        if t and not service.may_undo(s, member, task=t):
            return [service.out(member.id, f"Only the person who asked for <b>{html.escape(title)}</b>, its owner or the lead can remove it.", "system")]
        return [service.out(member.id, f"↩️ Removed <b>{html.escape(title)}</b>.", "system")] if service.delete_task(s, parts[1]) else \
            [service.out(member.id, "That task is already gone.", "system")]
    if kind == "mc":                                      # cancel a meeting the agent scheduled from conversation
        a = s.get(Appointment, int(parts[1]))
        if a and not service.may_undo(s, member, appointment=a):
            return [service.out(member.id, "Only the person who set up this meeting or the lead can cancel it.", "system")]
        return [service.out(member.id, "↩️ Meeting cancelled.", "system")] if service.cancel_appointment(s, int(parts[1])) else \
            [service.out(member.id, "That meeting is already cancelled.", "system")]
    if kind == "ul":                                      # undo a dependency the agent added
        t = s.get(Task, parts[1])
        if t and not service.may_undo(s, member, task=t):
            return [service.out(member.id, "Only the task's owner, the person who asked for it or the lead can remove that dependency.", "system")]
        return [service.out(member.id, "↩️ Dependency removed.", "system")] if service.remove_dependency(s, parts[1], parts[2]) else \
            [service.out(member.id, "That dependency is already gone.", "system")]
    return []


def link(s: Session, member: Member, channel: str, **ids) -> None:
    """Attach a platform identity to a member; the first app they connect becomes preferred."""
    for k, v in ids.items():
        setattr(member, k, v)
    if not member.preferred_channel:
        member.preferred_channel = channel
    s.add(member)
    s.commit()
    bus.publish("board")


def status() -> dict:
    return {name: ch.ready() for name, ch in REGISTRY.items()}
