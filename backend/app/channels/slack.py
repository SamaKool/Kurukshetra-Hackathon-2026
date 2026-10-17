"""Slack via Bolt for Python (async).

Dev: Socket Mode (SLACK_APP_TOKEN), so no public URL or tunnel is needed.
Prod: Slack calls POST /api/slack/events on the ALB; Bolt verifies the signing secret.

Members are linked automatically by email (users:read.email scope); DMs map to handle_dm,
messages in SLACK_PROJECT_CHANNEL map to handle_group (conflict detection), and Block Kit
button values carry the same callback strings Telegram uses.
"""
import logging
import re

from sqlmodel import Session, select

from .. import channels, service
from ..config import settings
from ..db import engine
from ..models import Member
from ..taskparse import is_task_command

log = logging.getLogger("slack")


def blocks_for(item: dict) -> list[dict]:
    blocks: list[dict] = [{"type": "section", "text": {"type": "mrkdwn", "text": channels.to_slack(item["text"])}}]
    for r, row in enumerate(item.get("buttons") or []):
        blocks.append({"type": "actions", "block_id": f"coord_row_{r}", "elements": [
            {"type": "button", "action_id": f"coord_{r}_{i}", "value": data,
             "text": {"type": "plain_text", "text": label[:75], "emoji": True},
             **({"style": "primary"} if label.startswith(("✅", "👍")) else {})}
            for i, (label, data) in enumerate(row)]})
    return blocks


_CHANNEL_ID = re.compile(r"^[CGD][A-Z0-9]{8,}$")


class SlackChannel:
    name = "slack"

    def __init__(self):
        self.app = None
        self.handler = None            # FastAPI request handler (HTTP mode)
        self.socket = None             # Socket Mode handler (dev)
        self.bot_user_id: str | None = None

    def ready(self) -> bool:
        return self.app is not None

    def linked(self, m: Member) -> bool:
        return bool(m.slack_dm_id)

    async def send(self, m: Member, item: dict) -> str | None:
        r = await self.app.client.chat_postMessage(channel=m.slack_dm_id, text=channels.to_plain(item["text"]),
                                                   blocks=blocks_for(item))
        return r["ts"]

    async def post(self, channel: str, text: str, blocks: list[dict] | None, thread_ts: str | None = None) -> str | None:
        """Post to a channel (meeting summaries, the agent's group messages, alerts)."""
        if not self.app or not channel:
            return None
        r = await self.app.client.chat_postMessage(channel=channel, text=text, blocks=blocks or None, thread_ts=thread_ts,
                                                   unfurl_links=False)
        return r["ts"]

    # ------------------------------------------------------------ lifecycle
    def build(self) -> None:
        if not settings.slack_enabled or self.app is not None:
            return
        from slack_bolt.async_app import AsyncApp

        self.app = AsyncApp(token=settings.slack_bot_token,
                            signing_secret=settings.slack_signing_secret or "unused-in-socket-mode",
                            request_verification_enabled=bool(settings.slack_signing_secret))
        _register(self.app)
        if settings.slack_signing_secret:
            from slack_bolt.adapter.fastapi.async_handler import (
                AsyncSlackRequestHandler,
            )
            self.handler = AsyncSlackRequestHandler(self.app)

    async def resolve_channels(self) -> None:
        """Let .env say `#proj-coordinator` instead of an opaque ID. Needs the channels:read (and groups:read, for
        private channels) scopes; without them, put the channel IDs in .env."""
        names = {a: getattr(settings, a) for a in ("slack_project_channel", "slack_summary_channel", "slack_alert_channel")
                 if getattr(settings, a) and not _CHANNEL_ID.match(getattr(settings, a))}
        if not names:
            return
        found: dict[str, str] = {}
        try:
            cursor = None
            while True:
                r = await self.app.client.conversations_list(types="public_channel,private_channel", limit=200,
                                                             exclude_archived=True, cursor=cursor)
                found.update({c["name"]: c["id"] for c in r["channels"]})
                cursor = (r.get("response_metadata") or {}).get("next_cursor")
                if not cursor:
                    break
        except Exception as e:
            log.warning("cannot look up Slack channel names (%s): use channel IDs in .env, or grant channels:read "
                        "and groups:read and reinstall the app", e)
            return
        for attr, name in names.items():
            cid = found.get(name.lstrip("#").lower())
            if cid:
                setattr(settings, attr, cid)
                log.info("Slack %s = #%s (%s)", attr.removeprefix("slack_"), name.lstrip("#"), cid)
            else:
                log.warning("Slack channel %r not found: create it and /invite the bot (private channels need an invite)", name)

    async def start(self, socket_mode: bool) -> None:
        self.build()
        if not self.app:
            return
        try:
            auth = await self.app.client.auth_test()
            self.bot_user_id = auth.get("user_id")
        except Exception as e:
            log.warning("Slack auth_test failed: %s", e)
        await self.resolve_channels()
        if socket_mode and settings.slack_app_token:
            from slack_bolt.adapter.socket_mode.async_handler import (
                AsyncSocketModeHandler,
            )
            self.socket = AsyncSocketModeHandler(self.app, settings.slack_app_token)
            await self.socket.connect_async()
            log.info("Slack connected via Socket Mode")
        else:
            log.info("Slack ready (HTTP events at /api/slack/events)")

    async def stop(self) -> None:
        if self.socket:
            await self.socket.close_async()


CHANNEL = SlackChannel()
channels.register(CHANNEL)


async def _resolve_member(s: Session, client, user_id: str, dm_channel: str | None) -> Member | None:
    m = s.exec(select(Member).where(Member.slack_user_id == user_id)).first()
    if m:
        if dm_channel and m.slack_dm_id != dm_channel:
            channels.link(s, m, "slack", slack_dm_id=dm_channel)
        return m
    try:                                            # auto-link by email
        info = await client.users_info(user=user_id)
        email = (info["user"].get("profile", {}).get("email") or "").lower()
    except Exception:
        email = ""
    if email:
        m = s.exec(select(Member).where(Member.email == email)).first()
        if m:
            if not dm_channel:
                dm_channel = (await client.conversations_open(users=user_id))["channel"]["id"]
            channels.link(s, m, "slack", slack_user_id=user_id, slack_dm_id=dm_channel)
            return m
    return None


def _register(app) -> None:
    @app.event("message")
    async def on_message(event, client, say):
        if event.get("bot_id") or event.get("subtype"):
            return
        text, user, ch = event.get("text", ""), event.get("user"), event.get("channel")
        is_dm = event.get("channel_type") == "im"
        with Session(engine) as s:
            me = await _resolve_member(s, client, user, ch if is_dm else None)
            if me is None:
                if is_dm:
                    free = [x for x in s.exec(select(Member)).all() if not x.slack_user_id]
                    await say(text="Which team member are you?", blocks=blocks_for({
                        "text": "Hi! Which team member are you?",
                        "buttons": [[(x.name, f"iam:{x.key}")] for x in free]}))
                return
            text = re.sub(r"<@[A-Z0-9]+>", "", text).strip()
            if not is_dm and is_task_command(text) and (not settings.slack_project_channel or ch == settings.slack_project_channel):
                outbox = await service.task_command(s, me, text, channel="slack", ext_id=event.get("ts"))
                for item in [i for i in outbox if i["member_id"] == me.id]:          # answer in the channel thread
                    await say(text=channels.to_slack(item["text"]), blocks=blocks_for(item), thread_ts=event.get("ts"))
                await channels.deliver(s, [i for i in outbox if i["member_id"] != me.id])
                return
            if is_dm:
                outbox = await service.handle_dm(s, me, text, reply_to=event.get("thread_ts"), ext_id=event.get("ts"),
                                                 channel="slack")
            elif not settings.slack_project_channel or ch == settings.slack_project_channel:
                outbox = await service.handle_group(s, me, text, ext_id=event.get("ts"), channel="slack",
                                                    reply_to=event.get("thread_ts"))
            else:
                return
            await channels.deliver(s, outbox)

    @app.action(re.compile(r"^coord_"))
    async def on_button(ack, body, client):
        await ack()
        data = body["actions"][0]["value"]
        user = body["user"]["id"]
        with Session(engine) as s:
            dm = body.get("channel", {}).get("id")
            if data.startswith("iam:"):
                m = s.exec(select(Member).where(Member.key == data[4:])).first()
                if m:
                    channels.link(s, m, "slack", slack_user_id=user, slack_dm_id=dm)
                    await client.chat_postMessage(channel=dm, text=f"✅ Connected as {m.name}.")
                return
            me = await _resolve_member(s, client, user, dm)
            if me is None:
                return
            outbox = await channels.handle_callback(s, me, data)
            msg = body.get("message")
            if msg and dm:                     # remove buttons so they can't be tapped twice
                kept = [b for b in msg.get("blocks", []) if b.get("type") != "actions"]
                try:
                    await client.chat_update(channel=dm, ts=msg["ts"], blocks=kept, text=msg.get("text", ""))
                except Exception:
                    pass
            await channels.deliver(s, outbox)

    @app.command("/coordinator")
    async def on_command(ack, command, client, respond):
        await ack()
        sub = (command.get("text") or "status").strip().split()[0]
        with Session(engine) as s:
            if sub == "task":
                me = await _resolve_member(s, client, command["user_id"], None)
                if not me:
                    await respond("I do not know who you are yet: send me a direct message once so I can match your email.")
                    return
                outbox = await service.task_command(s, me, "task: " + (command.get("text") or "").strip()[4:], channel="slack")
                for item in [i for i in outbox if i["member_id"] == me.id]:
                    await respond(channels.to_slack(item["text"]))
                await channels.deliver(s, [i for i in outbox if i["member_id"] != me.id])
                return
            if sub == "checkin":
                me = await _resolve_member(s, client, command["user_id"], None)
                if not me or me.role != "lead":
                    await respond("Only the team lead can run check-ins.")
                    return
                outbox = service.followups(s, force=True)
                await channels.deliver(s, outbox)
                await respond(f"Sent {len(outbox)} check-in(s).")
                return
            b = service.board(s)
        lines = [f"{service.STATUS_EMOJI[t['status']]} *{t['title']}* · {t['owner']}"
                 + (f"\n      ⛔ {t['blocker_text']}" if t['status'] == "Blocked" and t['blocker_text'] else "")
                 for t in b["tasks"]]
        await respond("*Project status*\n" + "\n".join(lines))
