"""BTVN#3: mock flight booking agents with LangChain tools and LangGraph control.

Shared mock data, tools and harness for the three independent strategy files.
No real ticket or payment is created. The default model is deterministic.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool


DATE = "2026-10-07"
PASSENGER = "Nguyen Van A"


@dataclass(frozen=True)
class FlightConstraints:
    origin: str
    destination: str
    date: str
    depart_before: str = "12:00"
    max_price: int = 2_000_000

    def validate(self, flight: dict) -> tuple[bool, str]:
        for key in ("origin", "destination", "date"):
            if flight.get(key) != getattr(self, key):
                return False, f"{key} không khớp yêu cầu"
        if flight.get("depart_time", "99:99") >= self.depart_before:
            return False, "Giờ bay vượt giới hạn"
        if flight.get("price", float("inf")) > self.max_price:
            return False, "Giá vé vượt ngân sách"
        return True, "Hợp lệ"


class FlightStore:
    """Database in memory; every scenario gets its own isolated instance."""

    def __init__(self):
        self.flights = {
            "VN122": dict(flight_id="VN122", origin="SGN", destination="DAD", date=DATE, depart_time="08:10", price=1_850_000, airline="Vietnam Airlines", refundable=True),
            "QH118": dict(flight_id="QH118", origin="SGN", destination="DAD", date=DATE, depart_time="15:40", price=1_640_000, airline="Bamboo Airways", refundable=True),
            "VJ602": dict(flight_id="VJ602", origin="SGN", destination="DAD", date=DATE, depart_time="09:30", price=1_450_000, airline="Vietjet", refundable=False),
            "VN128": dict(flight_id="VN128", origin="SGN", destination="DAD", date=DATE, depart_time="11:15", price=2_450_000, airline="Vietnam Airlines", refundable=True),
        }
        self.seats = {fid: ["12A", "12B"] for fid in self.flights}
        self.bookings: dict[str, dict] = {}

    def search_flights(self, origin: str, destination: str, date: str, order: list[str] | None = None) -> dict:
        ids = order or list(self.flights)
        flights = [self.flights[fid].copy() for fid in ids if fid in self.flights and self.flights[fid]["origin"] == origin and self.flights[fid]["destination"] == destination and self.flights[fid]["date"] == date]
        return {"flights": flights, "count": len(flights)}

    def check_seat(self, flight_id: str) -> dict:
        if flight_id not in self.flights:
            return {"error": "flight_not_found"}
        return {"flight_id": flight_id, "available_seats": list(self.seats[flight_id])}

    def book_seat(self, flight_id: str, seat_number: str, passenger_name: str) -> dict:
        if flight_id not in self.flights or seat_number not in self.seats[flight_id]:
            return {"error": "seat_unavailable"}
        self.seats[flight_id].remove(seat_number)
        code = f"BK-{len(self.bookings) + 1:04d}"
        booking = {**self.flights[flight_id], "seat_number": seat_number, "passenger_name": passenger_name,
                   "booking_code": code, "status": "held", "is_paid": False}
        self.bookings[code] = booking
        return booking.copy()

    def pay(self, booking_code: str) -> dict:
        booking = self.bookings.get(booking_code)
        if not booking:
            return {"error": "booking_not_found"}
        booking["status"] = "confirmed"
        booking["is_paid"] = True
        return booking.copy()

    def get_booking(self, booking_code: str) -> dict:
        return self.bookings.get(booking_code, {"error": "booking_not_found"}).copy()


def flight_tools(store: FlightStore, order: list[str]):
    """Five real LangChain StructuredTools bound to the same mock store."""

    @tool
    def search_flights(origin: str, destination: str, date: str) -> dict:
        """Find flights for an origin, destination, and ISO date."""
        return store.search_flights(origin, destination, date, order)

    @tool
    def check_seat(flight_id: str) -> dict:
        """List available seats for a flight."""
        return store.check_seat(flight_id)

    @tool
    def book_seat(flight_id: str, seat_number: str, passenger_name: str) -> dict:
        """Hold an available seat; no payment is made."""
        return store.book_seat(flight_id, seat_number, passenger_name)

    @tool
    def pay(booking_code: str) -> dict:
        """Simulate payment for a held booking."""
        return store.pay(booking_code)

    @tool
    def get_booking(booking_code: str) -> dict:
        """Independently read the current booking state."""
        return store.get_booking(booking_code)

    return {t.name: t for t in (search_flights, check_seat, book_seat, pay, get_booking)}


class CompletionCriteria:
    def __init__(self, constraints: FlightConstraints):
        self.constraints = constraints

    def check(self, booking: dict) -> tuple[bool, dict]:
        flight_ok, reason = self.constraints.validate(booking)
        checks = {"exists": "booking_code" in booking, "confirmed": booking.get("status") == "confirmed",
                  "paid": booking.get("is_paid") is True, "flight_valid": flight_ok,
                  "known_flight": booking.get("flight_id") in {"VN122", "QH118", "VJ602", "VN128"}}
        return all(checks.values()), {**checks, "reason": reason}


class PermissionChecker:
    def check(self, action: str, flight: dict, approved: bool) -> tuple[bool, str]:
        if approved:
            return True, "Đã được phê duyệt"
        if action == "pay":
            return False, "Thanh toán cần người phê duyệt"
        if action == "book_seat" and (flight.get("price", 0) > 1_500_000 or not flight.get("refundable", True)):
            return False, "Giá vượt hạn mức tự duyệt hoặc vé không hoàn hủy"
        return True, "Trong quyền hạn"


class LoopDetector:
    def __init__(self, window: int = 6):
        self.actions = deque(maxlen=window)
        self.observations = deque(maxlen=window)
        self.last_progress: int | None = None
        self.stall = 0

    def action_warning(self, name: str, args: dict) -> str | None:
        fingerprint = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        if fingerprint in self.actions:
            return "LOOP: lặp cùng tool và tham số"
        self.actions.append(fingerprint)
        return None

    def observation_warning(self, value: dict, progress: int) -> str | None:
        fingerprint = json.dumps(value, sort_keys=True, ensure_ascii=False)
        self.observations.append(fingerprint)
        if self.observations.count(fingerprint) >= 3:
            return "LOOP: observation không đổi"
        self.stall = self.stall + 1 if self.last_progress == progress else 0
        self.last_progress = progress
        if self.stall >= 3:
            return "STALL: không có tiến triển"
        return None


class GroundingChecker:
    FACTS = re.compile(
        r"\b(?:[A-Z]{2}\d{3}|BK-[A-Z0-9]+|\d{1,2}[A-F]|\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2})\b"
        r"|\b\d{1,3}(?:\.\d{3})+đ"
    )

    def check(self, answer: str, observations: list[dict]) -> tuple[bool, list[str]]:
        source = " ".join(json.dumps(o, ensure_ascii=False) for o in observations)
        missing = [fact for fact in dict.fromkeys(self.FACTS.findall(answer)) if fact not in source]
        return not missing, missing


def handoff(reason: str, attempts: list[dict], state: dict, question: str) -> dict:
    return {"stop_reason": reason, "da_thu": attempts.copy(), "trang_thai": state.copy(), "cau_hoi_cho_nguoi": question}


SCENARIOS = {
    1: {"name": "Happy path", "order": ["VN122", "VJ602", "QH118", "VN128"], "approved": True},
    2: {"name": "Giờ bay sai, cần đổi hướng", "order": ["QH118", "VN122", "VJ602", "VN128"], "approved": True},
    3: {"name": "Cần phê duyệt", "order": ["VJ602"], "approved": False},
    4: {"name": "Không có chuyến bay", "order": [], "approved": False},
    5: {"name": "Bịa đặt dữ kiện", "order": [], "approved": False},
}


class MockFlightModel(BaseChatModel):
    """Deterministic BaseChatModel: emits LangChain tool calls from state snapshots."""

    @property
    def _llm_type(self) -> str:
        return "se373-flight-mock"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "MockFlightModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> ChatResult:
        request = json.loads(str(messages[-1].content))
        if request["task"] == "plan":
            names = ["search_flights", "check_seat", "book_seat", "pay", "get_booking"]
            phase_to_index = {"search": 0, "check": 1, "book": 2, "pay": 3, "verify": 4}
            ai = AIMessage(content=json.dumps(names[phase_to_index[request.get("phase", "search")]:]))
        elif request["task"] == "hallucinate":
            ai = AIMessage(content="Đã đặt VN999 ghế 5C giá 1.200.000đ.")
        else:
            phase = request["phase"]
            args: dict[str, Any]
            if phase == "search":
                name, args = "search_flights", {"origin": request["origin"], "destination": request["destination"], "date": request["date"]}
            elif phase == "check":
                candidates = [f for f in request["candidates"] if f not in request["excluded"]]
                if not candidates:
                    ai = AIMessage(content="Không còn chuyến bay khả dụng.")
                    return ChatResult(generations=[ChatGeneration(message=ai)])
                name, args = "check_seat", {"flight_id": candidates[0]}
            elif phase == "book":
                name, args = "book_seat", {"flight_id": request["selected"], "seat_number": request["seat"], "passenger_name": PASSENGER}
            elif phase == "pay":
                name, args = "pay", {"booking_code": request["booking_code"]}
            else:
                name, args = "get_booking", {"booking_code": request["booking_code"]}
            ai = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "mock_call", "type": "tool_call"}])
        return ChatResult(generations=[ChatGeneration(message=ai)])


@dataclass
class Session:
    strategy: str
    scenario: int
    store: FlightStore
    constraints: FlightConstraints
    approved: bool = False
    model: MockFlightModel = field(default_factory=MockFlightModel)
    status: str = "RUNNING"
    phase: str = "search"
    candidates: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    selected: str | None = None
    seat: str | None = None
    booking_code: str | None = None
    verified_booking: dict | None = None
    observations: list[dict] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    handoff_report: dict | None = None
    answer: str = ""
    model_calls: int = 0
    tool_calls: int = 0
    estimated_tokens: int = 0
    safety_catches: int = 0
    replans: int = 0
    plan: list[str] = field(default_factory=list)
    plan_index: int = 0
    detector: LoopDetector = field(default_factory=LoopDetector)
    trace: list[dict] = field(default_factory=list)

    def ask_model(self, request: dict) -> AIMessage:
        prompt = json.dumps(request, ensure_ascii=False)
        response = self.model.invoke([HumanMessage(content=prompt)])
        self.model_calls += 1
        self.estimated_tokens += (len(prompt) + len(str(response.content))) // 4
        return response

    def stop(self, status: str, reason: str, question: str) -> None:
        self.status = status
        self.handoff_report = handoff(reason, self.attempts, {"phase": self.phase, "selected": self.selected,
                                   "booking_code": self.booking_code, "tool_calls": self.tool_calls}, question)

def _request(session: Session) -> dict:
    return {"task": "act", "phase": session.phase, "origin": session.constraints.origin,
            "destination": session.constraints.destination, "date": session.constraints.date,
            "candidates": session.candidates, "excluded": session.excluded,
            "selected": session.selected, "seat": session.seat, "booking_code": session.booking_code}


def _business_progress(session: Session) -> int:
    """Count completed booking milestones, independent of observation count."""
    if session.verified_booking:
        return 5
    if session.phase == "verify":
        return 4
    if session.booking_code:
        return 3
    if session.selected and session.seat:
        return 2
    if session.candidates:
        return 1
    return 0


def execute(session: Session, action: dict, tools: dict, policy: Any) -> None:
    """Apply shared safety checks and dispatch recovery to the design policy."""
    name, args = action["name"], action["args"]
    warning = session.detector.action_warning(name, args)
    if warning:
        session.trace.append({"tool": name, "args": args, "outcome": "blocked", "reason": warning})
        session.safety_catches += 1
        session.stop("HANDOFF", warning, "Có thể thay đổi chặng bay hoặc ngày đi không?")
        return
    session.attempts.append({"tool": name, "args": args})
    flight = session.store.flights.get(args.get("flight_id") or session.selected or "", {})
    if name == "book_seat":
        valid, reason = session.constraints.validate(flight)
        if not valid:
            session.safety_catches += 1
            event = {"tool": name, "args": args, "outcome": "blocked", "reason": reason}
            session.trace.append(event)
            previous_replans = session.replans
            policy.on_invalid_flight(session, args, reason)
            if session.replans > previous_replans:
                event["recovery"] = f"Replan #{session.replans}: {' -> '.join(session.plan)}"
            elif session.status == "RUNNING":
                event["recovery"] = "Reject flight and choose another candidate from observations."
            return
    if name in {"book_seat", "pay"}:
        allowed, reason = PermissionChecker().check(name, flight, session.approved)
        if not allowed:
            session.trace.append({"tool": name, "args": args, "outcome": "blocked", "reason": reason})
            session.safety_catches += 1
            session.stop("APPROVAL_REQUIRED", reason, "Bạn có phê duyệt thao tác này không?")
            return

    result = tools[name].invoke(args)
    if isinstance(result, str):
        result = json.loads(result)
    session.trace.append({"tool": name, "args": args,
                          "outcome": "error" if "error" in result else "ok", "result": result})
    session.tool_calls += 1
    session.observations.append(result)
    if name == "search_flights":
        session.candidates = [flight["flight_id"] for flight in result["flights"]]
        if not session.candidates:
            policy.on_no_flights(session)
            if session.status != "RUNNING":
                return
        session.phase = "check" if session.candidates else "search"
    elif name == "check_seat":
        if not result.get("available_seats"):
            policy.on_no_seats(session, args)
            if session.status != "RUNNING":
                return
        else:
            session.selected = args["flight_id"]
            session.seat = result["available_seats"][0]
            session.phase = "book"
    elif name == "book_seat":
        if "error" in result:
            session.stop("HANDOFF", result["error"], "Có cần chọn ghế khác không?")
            return
        session.booking_code = result["booking_code"]
        session.phase = "pay"
    elif name == "pay":
        session.phase = "verify"
    elif name == "get_booking":
        passed, details = CompletionCriteria(session.constraints).check(result)
        if not passed:
            session.stop("HANDOFF", f"COMPLETION_FAILED: {details}", "Cần kiểm tra lại booking không?")
            return
        session.verified_booking = result
        session.answer = (
            f"Đã xác nhận {result['booking_code']} chuyến "
            f"{result['flight_id']} ghế {result['seat_number']}."
        )
        grounded, missing = GroundingChecker().check(session.answer, session.observations)
        if not grounded:
            session.safety_catches += 1
            session.stop("UNSUPPORTED_CLAIM", f"Không có căn cứ: {missing}", "Cần truy vấn booking lại không?")
            return
        session.status = "COMPLETED"
    if session.status == "RUNNING":
        warning = session.detector.observation_warning(result, _business_progress(session))
        if warning:
            session.safety_catches += 1
            session.stop("HANDOFF", warning, "Có cần điều chỉnh yêu cầu không?")
