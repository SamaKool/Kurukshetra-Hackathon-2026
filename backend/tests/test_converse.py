"""Ordinary conversation -> tasks, owners, deadlines and dependencies; nobody tags the bot or types a command."""
import asyncio
from datetime import date
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app import channels, converse, service
from app.channels import telegram as tg
from app.config import settings
from app.db import engine as db_engine
from app.main import app
from app.models import Member, Message, Task
from app.pipeline import llm
from app.textutil import looks_like_question, states_a_condition

TODAY = date(2026, 10, 17)                          # a Saturday


def run(coro):
    return asyncio.run(coro)


def m(i, name, key=None, username=None):
    return Member(id=i, key=key or name.lower(), name=name, role="member", tg_username=username)


def t(tid, title, owner=1, status="Pending", deps=""):
    return Task(id=tid, title=title, owner_id=owner, status=status, depends_on=deps, aliases=title.lower())


TEAM = [m(1, "Tejas", username="tejasnalawade"), m(2, "Soham", username="JustSoham"), m(3, "Samarth"), m(4, "Atharva")]
SPEAKER = TEAM[2]                                    # Samarth


def val(items, tasks=(), text=None, **kw):
    return converse.validate(items, SPEAKER, TEAM, list(tasks), TODAY, text=text, **kw)


def task_item(title, owner=None, conf=0.9, **extra):
    return {"kind": "task", "title": title, "owner": owner, "confidence": conf, "evidence": "x", **extra}


# ================================================================ the cheap gate (no model call for small talk)

@pytest.mark.parametrize("text", [
    "Soham please build the backend API by Friday", "I'll write the test cases after the dashboard is ready", "we need the report by end of day",
    "@JustSoham the deployment is stopped, fix it asap", "can you send me the .env file", "the dashboard depends on the backend", "kal tak kar do",
])
def test_messages_that_could_be_about_work_reach_the_model(text):
    assert converse.looks_actionable(text)


@pytest.mark.parametrize("text", ["lol", "ok", "haha nice one", "I like kids", "good morning team", "thanks!", "", "   ", "yes", "I grabbed a samosa"])
def test_small_talk_never_reaches_the_model(text):
    assert not converse.looks_actionable(text)


@pytest.mark.parametrize("text,q", [("Did anyone catch the cricket score", True), ("is the dashboard blocked on anything?", True),
                                    ("Samarth is the dashboard blocked", False), ("I'll send it", False), ("Will do", False)])
def test_question_detection(text, q):
    assert looks_like_question(text) is q


@pytest.mark.parametrize("text,c", [("deployment needs the API done first", True), ("once the API is done I start", True), ("the API is done", False),
                                    ("I finished it", False), ("wait until it is finished", True)])
def test_a_condition_is_not_a_claim(text, c):
    assert states_a_condition(text) is c


# ================================================================ validation: the model proposes, the rules decide

def test_a_confident_assignment_becomes_a_task_for_that_person():
    (p,) = val([task_item("Fix the deployment", "Soham", due="2026-10-18", estimate_minutes=90)])
    assert (p.kind, p.title, p.owner_id, p.due, p.est_minutes) == ("task", "Fix the deployment", 2, "2026-10-18", 90)


def test_low_confidence_guesses_are_dropped_not_acted_on():
    assert val([task_item("Maybe do something", "Soham", conf=0.69)]) == []
    assert len(val([task_item("Maybe do something", "Soham", conf=0.69)], min_confidence=0.6)) == 1


@pytest.mark.parametrize("owner,expected", [("Soham", 2), ("soham", 2), ("JustSoham", 2), ("tejasnalawade", 1), ("the speaker", 3), ("Samarth", 3), ("Ath", 4)])
def test_owners_resolve_by_name_telegram_handle_or_the_speaker(owner, expected):
    assert val([task_item("Do the thing", owner)])[0].owner_id == expected


@pytest.mark.parametrize("owner", [None, "", "null", "nobody", "Zoe", "S"])
def test_an_unknown_or_ambiguous_owner_falls_back_to_whoever_raised_it_never_to_an_invented_person(owner):
    assert val([task_item("Do the thing", owner)])[0].owner_id == SPEAKER.id


def test_dates_must_be_real_and_not_in_the_past():
    assert val([task_item("Do the thing", due="2026-10-25")])[0].due == "2026-10-25"
    assert val([task_item("Do the thing", due="2026-10-01")])[0].due is None
    assert val([task_item("Do the thing", due="soon")])[0].due is None
    assert val([task_item("Do the thing", due="2026-13-45")])[0].due is None


@pytest.mark.parametrize("est,expected", [(90, 90), (0, None), (-5, None), ("2h", None), (60 * 24 * 30, None), (None, None)])
def test_estimates_must_be_positive_and_sane(est, expected):
    assert val([task_item("Do the thing", estimate_minutes=est)])[0].est_minutes == expected


@pytest.mark.parametrize("title", ["", "ab", "x" * 91])
def test_titles_must_be_a_sensible_length(title):
    assert val([task_item(title)]) == []


def test_work_already_on_the_board_is_not_created_twice():
    board = [t("T1", "Build the backend API"), t("T2", "Write the test cases", status="Complete")]
    assert val([task_item("Build the backend API")], board) == []
    assert val([task_item("Backend API build")], board) == []                  # reworded
    assert val([task_item("Write the test cases")], board) == []               # even if it is already finished
    assert len(val([task_item("Build the dashboard")], board)) == 1


def test_prerequisites_by_existing_id_and_by_a_task_created_in_the_same_message():
    board = [t("T1", "Build the backend API")]
    (p,) = val([task_item("Build the dashboard", after=["T1"])], board)
    assert p.after == ["T1"]
    a, b = val([task_item("Do the design review", "Tejas"), task_item("Build the dashboard", after=["Do the design review"])], board)
    assert b.after == ["new:do the design review"]
    assert val([task_item("Build the dashboard", after=["T9", "nothing like this"])], board)[0].after == []      # unknown refs vanish


def test_a_relationship_is_only_accepted_if_the_message_actually_mentions_the_task():
    board = [t("T1", "Build the backend API"), t("T2", "Write the test cases")]
    dep = {"kind": "dependency", "task": "T2", "after": "T1", "confidence": 0.9, "evidence": "x"}
    assert val([dep], board, text="deployment needs the dashboard first") == []                                 # never mentions either
    assert val([dep], board, text="the test cases need the backend API first") != []
    assert val([task_item("Deploy it", after=["T1"])], board, text="deploy once the dashboard works")[0].after == []


def test_dependencies_need_two_different_real_tasks():
    board = [t("T1", "Build the backend API"), t("T2", "Build the dashboard")]
    ok = {"kind": "dependency", "task": "T2", "after": "T1", "confidence": 0.9}
    assert len(val([ok], board)) == 1
    assert val([{**ok, "after": "T2"}], board) == []                          # not itself
    assert val([{**ok, "after": "T7"}], board) == []                          # not a task that does not exist
    assert val([{**ok, "after": "the design review"}], board) == []           # not a phrase


def test_junk_from_the_model_is_ignored():
    assert val([{"kind": "banana"}, {"kind": "task"}, {"kind": "task", "title": "Fine title", "confidence": "high"}, "nope"]) == []   # type: ignore[list-item]


def test_at_most_a_few_changes_per_message():
    assert len(val([task_item(f"Distinct job number {w}") for w in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")])) == converse.MAX_ITEMS


# ================================================================ asking the model

def fake_llm(monkeypatch, answer):
    seen = []

    async def complete_json(system, user, max_tokens=4000):
        seen.append(user)
        return answer(user) if callable(answer) else answer
    monkeypatch.setattr(llm, "complete_json", complete_json)
    monkeypatch.setattr(type(settings), "llm_enabled", property(lambda self: True))
    return seen


def test_no_llm_means_no_changes_and_no_crash(monkeypatch):
    monkeypatch.setattr(type(settings), "llm_enabled", property(lambda self: False))
    assert run(converse.ask(SPEAKER, "Soham please build the API", TEAM, [], TODAY)) == []


@pytest.mark.parametrize("answer", [None, {}, {"items": None}, {"items": "x"}, {"items": [1, "a"]}, ["list"], "text"])
def test_a_garbage_model_answer_means_nothing_happens(monkeypatch, answer):
    fake_llm(monkeypatch, answer)
    assert run(converse.ask(SPEAKER, "Soham please build the API", TEAM, [], TODAY)) == []


def test_the_model_is_told_the_team_the_board_today_and_who_is_speaking(monkeypatch):
    seen = fake_llm(monkeypatch, {"items": []})
    run(converse.ask(SPEAKER, "Soham please build the API", TEAM, [t("T1", "Build the dashboard", 3)], TODAY, ["earlier line"]))
    ctx = seen[0]
    assert "TODAY: 2026-10-17 (Saturday)" in ctx and "TEAM: Tejas, Soham, Samarth, Atharva" in ctx and "SPEAKER: Samarth" in ctx
    assert "T1: Build the dashboard (owner Samarth, Pending)" in ctx and "earlier line" in ctx and "MESSAGE: Soham please build the API" in ctx


# ================================================================ dependencies in the database

@pytest.fixture()
def world(monkeypatch):
    monkeypatch.setattr(settings, "auto_tasks", True)
    with TestClient(app) as c:
        c.post("/api/reset")
        with Session(db_engine) as s:
            for x in s.exec(select(Member)).all():
                x.tg_user_id = x.tg_chat_id = x.slack_user_id = x.slack_dm_id = x.preferred_channel = x.tg_username = None
                s.add(x)
            s.commit()
        yield c


def key_id(key):
    with Session(db_engine) as s:
        return s.exec(select(Member).where(Member.key == key)).first().id


def deps_of(tid):
    with Session(db_engine) as s:
        return s.get(Task, tid).depends_on


def test_add_dependency_links_two_tasks(world):
    with Session(db_engine) as s:
        service.add_dependency(s, "T5", "T1")
    assert deps_of("T5") == "T4,T1"                                   # T5 already waited for T4


@pytest.mark.parametrize("a,b,why", [("T1", "T1", "itself"), ("T1", "T99", "unknown"), ("T99", "T1", "unknown"), ("T5", "T4", "already"),
                                     ("T3", "T2", "loop"), ("T3", "T4", "loop"), ("T3", "T5", "loop")])
def test_bad_dependencies_are_refused(world, a, b, why):
    with Session(db_engine) as s, pytest.raises(ValueError, match=why):
        service.add_dependency(s, a, b)


def test_remove_dependency(world):
    with Session(db_engine) as s:
        assert service.remove_dependency(s, "T5", "T4") is True
        assert service.remove_dependency(s, "T5", "T4") is False
        assert service.remove_dependency(s, "T99", "T1") is False
    assert deps_of("T5") == ""


def test_dependency_api(world):
    assert world.post("/api/tasks/T1/dependencies", json={"depends_on": "T3"}).status_code == 200
    assert deps_of("T1") == "T3"
    assert world.post("/api/tasks/T3/dependencies", json={"depends_on": "T1"}).status_code == 400       # a loop
    assert world.post("/api/tasks/T99/dependencies", json={"depends_on": "T1"}).status_code == 404
    assert world.delete("/api/tasks/T1/dependencies/T3").status_code == 200
    assert world.delete("/api/tasks/T1/dependencies/T3").status_code == 404


def test_the_board_exposes_prerequisite_ids_for_editing(world):
    t5 = next(x for x in world.get("/api/board").json()["tasks"] if x["id"] == "T5")
    assert t5["depends_on"] == "T4"


# ================================================================ the group flow

def say_in_group(key, text):
    with Session(db_engine) as s:
        me = s.exec(select(Member).where(Member.key == key)).first()
        return run(service.handle_group(s, me, text, channel="telegram"))


def board():
    with Session(db_engine) as s:
        return {x.id: x for x in s.exec(select(Task)).all()}


def test_an_assignment_in_plain_words_creates_a_task_with_owner_estimate_and_deadline(world, monkeypatch):
    fake_llm(monkeypatch, {"items": [task_item("Fix the deployment", "Priya", due="2099-01-01", estimate_minutes=60)]})
    out = say_in_group("kabir", "priya the deployment is stopped, fix it asap")
    new = [x for x in board().values() if x.title == "Fix the deployment"]
    assert len(new) == 1 and new[0].owner_id == key_id("priya") and new[0].est_minutes == 60 and new[0].deadline == "2099-01-01"
    assert len(out) == 1 and out[0]["member_id"] == key_id("kabir") and "🆕" in out[0]["text"] and "Fix the deployment" in out[0]["text"]
    assert f"{{M{key_id('priya')}}}" in out[0]["text"]                                      # the assignee is @mentioned, not just named
    assert out[0]["buttons"] == [[(f"↩️ Undo {new[0].id}", f"dt:{new[0].id}")]]


def test_a_stated_dependency_is_written_to_the_task(world, monkeypatch):
    fake_llm(monkeypatch, {"items": [{"kind": "dependency", "task": "T1", "after": "T3", "confidence": 0.9, "evidence": "x"}]})
    out = say_in_group("aanya", "the slides can't start until the sandbox credentials are ready")
    assert deps_of("T1") == "T3" and "🔗" in out[0]["text"] and out[0]["buttons"] == [[("↩️ Unlink T1", "ul:T1:T3")]]


def test_a_task_and_its_prerequisite_can_arrive_in_one_message(world, monkeypatch):
    fake_llm(monkeypatch, {"items": [task_item("Do the design review", "Aanya"),
                                      {"kind": "dependency", "task": "T4", "after": "Do the design review", "confidence": 0.9}]})
    say_in_group("meera", "the checkout page must wait for the design review that Aanya does")
    review = next(x for x in board().values() if x.title == "Do the design review")
    assert review.owner_id == key_id("aanya") and review.id in deps_of("T4").split(",")


def test_small_talk_costs_nothing_and_gets_no_reply(world, monkeypatch):
    seen = fake_llm(monkeypatch, {"items": [task_item("Should never be created")]})
    for text in ("lol", "ok", "good morning", "I grabbed a samosa"):
        assert say_in_group("kabir", text) == []
    assert seen == [] and not any(x.title == "Should never be created" for x in board().values())


def test_when_the_model_finds_nothing_the_bot_stays_quiet(world, monkeypatch):
    fake_llm(monkeypatch, {"items": []})
    assert say_in_group("kabir", "can someone call me tomorrow please") == []


def test_the_whole_feature_can_be_switched_off(world, monkeypatch):
    seen = fake_llm(monkeypatch, {"items": [task_item("Fix the deployment", "Priya")]})
    monkeypatch.setattr(settings, "auto_tasks", False)
    assert say_in_group("kabir", "priya the deployment is stopped, fix it asap") == [] and seen == []


def test_a_status_report_that_names_no_task_is_about_the_speakers_only_open_task(world):
    out = say_in_group("priya", "tested, all green, PR #14")                       # Priya owns only T3, which waits for nothing
    assert board()["T3"].status == "Complete" and out


def test_with_several_open_tasks_the_bot_asks_which_one(world):
    with Session(db_engine) as s:
        service.create_task(s, title="Prepare the pitch", owner_id=key_id("priya"))
    out = say_in_group("priya", "done ✅ pushed to main")
    assert board()["T3"].status == "In Progress" and out[0]["text"] == "Which task is this about?"
    assert {b[0][1] for b in out[0]["buttons"]} == {"pick:T3", "pick:T6"}


def test_a_question_is_not_the_speakers_own_status(world):
    assert say_in_group("kabir", "is the payment api blocked on anything?") == []
    say_in_group("priya", "Meera is the checkout blocked on anything")
    assert board()["T2"].status == "Pending" and board()["T3"].status == "In Progress" and board()["T4"].status == "In Progress"


def test_a_message_about_someone_elses_work_is_not_my_status_report(world):
    say_in_group("priya", "kabir the payment api is blocked, waiting for the credentials")
    assert board()["T3"].status == "In Progress"                                  # Priya's own task is untouched


def test_needs_done_first_is_not_a_claim_that_something_is_done(world):
    assert say_in_group("meera", "the checkout needs the payment api done first") == []


def test_a_real_claim_is_still_challenged(world):
    out = say_in_group("meera", "integration's done, starting checkout now")
    assert {o["member_id"] for o in out} == {key_id("meera"), key_id("kabir")}


def test_chit_chat_about_a_task_is_not_answered_in_the_group(world):
    out = say_in_group("kabir", "the payment api lol")
    assert out == []


# ================================================================ undo buttons

def press(key, data):
    with Session(db_engine) as s:
        me = s.exec(select(Member).where(Member.key == key)).first()
        return run(channels.handle_callback(s, me, data))


def test_undo_removes_a_task_the_bot_created(world, monkeypatch):
    fake_llm(monkeypatch, {"items": [task_item("Fix the deployment", "Priya")]})
    say_in_group("kabir", "priya the deployment is stopped, fix it asap")
    tid = next(x.id for x in board().values() if x.title == "Fix the deployment")
    out = press("kabir", f"dt:{tid}")
    assert tid not in board() and "Removed" in out[0]["text"]
    assert "already gone" in press("kabir", f"dt:{tid}")[0]["text"]


def test_unlink_removes_a_dependency_the_bot_added(world):
    with Session(db_engine) as s:
        service.add_dependency(s, "T1", "T3")
    assert "removed" in press("kabir", "ul:T1:T3")[0]["text"] and deps_of("T1") == ""
    assert "already gone" in press("kabir", "ul:T1:T3")[0]["text"]


# ================================================================ Telegram: @usernames and group replies

def tg_update(user_id, username=None, chat_type="supergroup", text=""):
    msg = SimpleNamespace(text=text)
    return SimpleNamespace(effective_user=SimpleNamespace(id=user_id, username=username, first_name="X"), effective_chat=SimpleNamespace(id=-1001, type=chat_type), message=msg)


def test_the_handle_people_are_tagged_by_is_learned_when_they_write(world):
    with Session(db_engine) as s:
        me = s.exec(select(Member).where(Member.key == "priya")).first()
        me.tg_user_id = 777
        s.add(me)
        s.commit()
        assert tg._member(s, tg_update(777, "PriyaP")).key == "priya"
        assert s.exec(select(Member).where(Member.key == "priya")).first().tg_username == "PriyaP"
        tg._member(s, tg_update(777, "PriyaNew"))                              # and it follows a rename
        assert s.exec(select(Member).where(Member.key == "priya")).first().tg_username == "PriyaNew"


@pytest.mark.parametrize("text,expected", [
    ("@JustSoham the deployment is stopped", "Kabir the deployment is stopped"),
    ("@justsoham fix it", "Kabir fix it"),                                          # case-insensitive
    ("@xdominator_bot  please help", "please help"),                                # the bot's own tag is dropped
    ("hey @someone_else look", "hey @someone_else look"),                           # unknown handles are left alone
    ("no tags here", "no tags here"),
])
def test_tagged_handles_become_team_names(world, text, expected):
    with Session(db_engine) as s:
        me = s.exec(select(Member).where(Member.key == "kabir")).first()
        me.tg_username = "JustSoham"
        s.add(me)
        s.commit()
        assert tg.resolve_mentions(s, text, "xdominator_bot") == expected


class FakeTG:
    name = "telegram"

    def __init__(self):
        self.sent = []

    def ready(self):
        return True

    def linked(self, member):
        return bool(member.tg_chat_id)

    async def send(self, member, item):
        self.sent.append(("dm", member.key, item["text"]))
        return "dm1"

    async def send_group(self, chat_id, html, markup=None, reply_to=None):
        self.sent.append(("group", chat_id, html, markup, reply_to))
        return "g1"


@pytest.fixture()
def tgchan(world, monkeypatch):
    fake = FakeTG()
    monkeypatch.setitem(channels.REGISTRY, "telegram", fake)
    with Session(db_engine) as s:
        for key, uid in (("kabir", 111), ("priya", 222)):
            x = s.exec(select(Member).where(Member.key == key)).first()
            x.tg_user_id = uid                                                # identified in the group, no private chat
            s.add(x)
        from app.models import Setting
        for row in s.exec(select(Setting)).all():
            s.delete(row)
        s.add(Setting(key="tg_group_chat_id", value="-1001"))
        s.commit()
    return fake


def deliver(items, **kw):
    with Session(db_engine) as s:
        return run(channels.deliver(s, items, **kw))


def item(key, text, buttons=None):
    return {"member_id": key_id(key), "text": text, "kind": "confirm", "task_id": None, "buttons": buttons or [], "reply_to": None}


def test_the_senders_answer_goes_into_the_group_as_a_reply_to_their_message(tgchan):
    deliver([item("kabir", "✅ Got it", [[("↩️ Undo", "undo:1")]])], group_reply={"member_id": key_id("kabir"), "chat_id": -1009, "message_id": 55})
    kind, chat, html, markup, reply_to = tgchan.sent[0]
    assert (kind, chat, html, reply_to) == ("group", -1009, "✅ Got it", 55) and markup == [[("↩️ Undo", "undo:1")]]


def test_someone_who_never_opened_a_private_chat_is_reached_in_the_group_with_a_mention(tgchan):
    deliver([item("priya", "Kabir is blocked on the API. Can you help?")])
    kind, chat, html, markup, reply_to = tgchan.sent[0]
    assert (kind, chat, reply_to) == ("group", -1001, None) and 'tg://user?id=222' in html and "Kabir is blocked" in html


def test_someone_with_a_private_chat_still_gets_a_dm(tgchan):
    with Session(db_engine) as s:
        x = s.exec(select(Member).where(Member.key == "priya")).first()
        x.tg_chat_id = 222
        s.add(x)
        s.commit()
    deliver([item("priya", "hello")])
    assert tgchan.sent[0][0] == "dm"


def test_other_people_are_not_pulled_into_a_senders_group_reply(tgchan):
    deliver([item("kabir", "for the sender"), item("priya", "for someone else")], group_reply={"member_id": key_id("kabir"), "chat_id": -1009, "message_id": 5})
    assert tgchan.sent[0][4] == 5 and tgchan.sent[1][4] is None and 'tg://user?id=222' in tgchan.sent[1][2]


def test_mention_tokens_in_a_message_become_real_mentions(tgchan):
    deliver([item("kabir", f"🆕 T9 · Fix it → {{M{key_id('priya')}}}")], group_reply={"member_id": key_id("kabir"), "chat_id": -1009, "message_id": 5})
    assert 'tg://user?id=222' in tgchan.sent[0][2] and "{M" not in tgchan.sent[0][2]


def test_unknown_people_without_any_route_are_still_recorded_not_lost(tgchan):
    deliver([item("meera", "nobody can reach Meera")])                                      # not identified anywhere
    with Session(db_engine) as s:
        rows = s.exec(select(Message).where(Message.direction == "out")).all()
    assert tgchan.sent == [] and rows and rows[-1].channel == "dev"


# ================================================================ inferring dependencies between existing tasks

def test_infer_links_only_strong_valid_relationships(world, monkeypatch):
    # open tasks in the sample project: T2 (waits for T3), T3, T4 (waits for T2), T5 (waits for T4); T1 is finished
    fake_llm(monkeypatch, {"items": [
        {"kind": "dependency", "task": "T4", "after": "T3", "confidence": 0.9, "evidence": "checkout needs the credentials"},   # allowed
        {"kind": "dependency", "task": "T3", "after": "T5", "confidence": 0.9, "evidence": "would create a loop"},              # refused
        {"kind": "dependency", "task": "T5", "after": "T3", "confidence": 0.5, "evidence": "weak"},                              # too weak
        {"kind": "dependency", "task": "T4", "after": "T99", "confidence": 0.99, "evidence": "unknown"},                         # not a task
        {"kind": "dependency", "task": "T1", "after": "T3", "confidence": 0.99, "evidence": "T1 is already finished"},           # not open
        {"kind": "dependency", "task": "T5", "after": "T4", "confidence": 0.99, "evidence": "already there"}]})
    with Session(db_engine) as s:
        linked = run(converse.infer_dependencies(s))
    assert linked == ["Build the checkout page now waits for Get sandbox test credentials"]
    assert deps_of("T4") == "T2,T3" and deps_of("T3") == "" and deps_of("T5") == "T4"


def test_infer_needs_an_llm(world, monkeypatch):
    monkeypatch.setattr(type(settings), "llm_enabled", property(lambda self: False))
    with Session(db_engine) as s, pytest.raises(ValueError, match="LLM"):
        run(converse.infer_dependencies(s))
    assert world.post("/api/tasks/infer-dependencies").status_code == 400


def test_infer_endpoint_reports_what_it_linked(world, monkeypatch):
    fake_llm(monkeypatch, {"items": [{"kind": "dependency", "task": "T4", "after": "T3", "confidence": 0.9}]})
    assert world.post("/api/tasks/infer-dependencies").json() == {"linked": ["Build the checkout page now waits for Get sandbox test credentials"]}


# ================================================================ the dev database gains new columns without being deleted

def test_a_new_nullable_column_is_added_to_an_existing_dev_database(tmp_path, monkeypatch):
    from sqlalchemy import create_engine, inspect, text

    from app import db
    eng = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE member (id INTEGER PRIMARY KEY, key TEXT, name TEXT, role TEXT)"))     # an old schema
    monkeypatch.setattr(db, "engine", eng)
    added = db.add_missing_columns()
    assert "member.tg_username" in added and "member.slack_dm_id" in added
    assert "tg_username" in {c["name"] for c in inspect(eng).get_columns("member")}
    assert db.add_missing_columns() == []                                                                       # idempotent


# ================================================================ /task in plain words: the model reads it, the strict form stays exact

def member_of(key):
    with Session(db_engine) as s:
        return s.exec(select(Member).where(Member.key == key)).first()


def task_cmd(key, text, answer, monkeypatch):
    seen = fake_llm(monkeypatch, answer)

    async def no_extract(*a, **k):
        return None
    monkeypatch.setattr(llm, "extract", no_extract)
    with Session(db_engine) as s:
        return service.task_command(s, s.get(Member, key_id(key)), text), seen


def said(outbox):
    return " ".join(i["text"] for i in outbox)


def test_a_plain_words_task_goes_to_the_named_person_with_a_clean_title(world, monkeypatch):
    coro, seen = task_cmd("kabir", "/task priya can you get the report from meera", {"items": [
        task_item("Get the report from Meera", "Priya", 0.6)]}, monkeypatch)
    box = run(coro)
    (t,) = [x for x in world.get("/api/board").json()["tasks"] if x["title"] == "Get the report from Meera"]
    assert t["owner"] == "Priya" and "T6" in said(box)
    assert "Design the slides (owner Aanya, Complete)" in seen[-1]          # finished work is visible to a /task too


def test_a_status_report_typed_as_a_task_updates_the_task_instead(world, monkeypatch):
    box = run(task_cmd("kabir", "/task done", {"items": [{"kind": "status", "task": "T2", "confidence": 0.9}]}, monkeypatch)[0])
    with Session(db_engine) as s:
        assert len(s.exec(select(Task)).all()) == 5                         # nothing new was created
        assert s.get(Task, "T2").status != "Pending" or "review" in said(box).lower()


def test_a_question_typed_as_a_task_is_answered_from_the_board(world, monkeypatch):
    box = run(task_cmd("kabir", "/task slides hue kya", {"items": [{"kind": "question", "task": "T1", "confidence": 0.9}]}, monkeypatch)[0])
    assert "T1" in said(box) and "Complete" in said(box) and "Aanya" in said(box)
    with Session(db_engine) as s:
        assert len(s.exec(select(Task)).all()) == 5


def test_work_already_on_the_board_is_pointed_at_not_created_twice(world, monkeypatch):
    box = run(task_cmd("kabir", "/task record the demo video", {"items": [task_item("Record the demo video", "Aanya")]}, monkeypatch)[0])
    assert "already on the board" in said(box) and "T5" in said(box)


def test_the_strict_form_never_asks_the_model(world, monkeypatch):
    coro, seen = task_cmd("kabir", "/task Write the docs @priya due friday", {"items": []}, monkeypatch)
    box = run(coro)
    assert seen == [] and "Created" in said(box)


def test_when_the_model_finds_nothing_the_strict_parser_still_answers(world, monkeypatch):
    box = run(task_cmd("kabir", "/task Clean up the CSS", {"items": []}, monkeypatch)[0])
    assert "Created" in said(box) and "Clean up the CSS" in said(box)


def test_earlier_messages_reach_the_model_with_who_said_them(world, monkeypatch):
    seen = fake_llm(monkeypatch, {"items": []})
    with Session(db_engine) as s:
        kabir, priya = s.get(Member, key_id("kabir")), s.get(Member, key_id("priya"))
        run(service.handle_group(s, priya, "the payment page looks good"))
        run(service.handle_group(s, kabir, "Priya please send me the staging password"))
    assert "Priya: the payment page looks good" in seen[-1]


# ================================================================ undo buttons are visible to everyone in a group

def test_only_the_requester_the_owner_or_the_lead_may_undo_a_task(world):
    with Session(db_engine) as s:
        kabir, priya, meera, aanya = (s.get(Member, key_id(k)) for k in ("kabir", "priya", "meera", "aanya"))
        t = service.create_task(s, title="Send the staging password", owner_id=priya.id, requested_by_id=kabir.id)
        assert service.may_undo(s, kabir, task=t) and service.may_undo(s, priya, task=t) and service.may_undo(s, aanya, task=t)
        assert not service.may_undo(s, meera, task=t)
        box = run(channels.handle_callback(s, meera, f"dt:{t.id}"))
        assert "Only the person" in box[0]["text"] and s.get(Task, t.id) is not None
        run(channels.handle_callback(s, kabir, f"dt:{t.id}"))
        assert s.get(Task, t.id) is None


# ================================================================ Telegram: recognised by name, no /join needed

def test_a_telegram_name_that_matches_one_team_member_links_them(world):
    u = SimpleNamespace(effective_user=SimpleNamespace(id=9001, first_name="KABIR SINGH", username=None),
                        effective_chat=SimpleNamespace(id=-1001, type="supergroup"))
    with Session(db_engine) as s:
        assert tg._member(s, u).key == "kabir"
    u2 = SimpleNamespace(effective_user=SimpleNamespace(id=9002, first_name="Pat", username="pat99"), effective_chat=u.effective_chat)
    with Session(db_engine) as s:
        assert tg._member(s, u2) is None and s.get(Member, key_id("kabir")).tg_user_id == 9001
