import sys
import traceback
from dataclasses import dataclass

from mlx_lm import generate, load
from PyQt5.QtCore import QObject, QThread, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from qt_material import apply_stylesheet


DEFAULT_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MAX_TOKENS = 4096
BRIEF_SYSTEM_PROMPT = """
You are a senior product strategist and Amazon working-backwards product writer.
When the user gives an idea, transform it into a clear product brief and an
Amazon-style PRD press release document.

Write in markdown. Be concrete, practical, and opinionated. If details are
missing, make reasonable assumptions and call them out briefly.

Always produce this structure:

# Product Brief
## One-Line Concept
## Target Customer
## Customer Problem
## Proposed Solution
## Core Use Cases
## Differentiation
## MVP Scope
## Success Metrics
## Risks and Open Questions

# Amazon-Style Press Release / PRD
## Press Release
Include a headline, subheadline, launch city/date placeholder, opening paragraph,
customer benefit paragraphs, a leadership quote, a customer quote, and availability.
## Customer FAQ
Include 6-8 customer-facing questions and answers.
## Internal FAQ
Include 6-8 internal product, business, technical, legal, and go-to-market questions.
""".strip()
PLAN_SYSTEM_PROMPT = """
You are a senior product manager and delivery planner.
When the user gives a product idea, create a roadmap that connects strategy to
execution. Write in markdown. Be concrete, realistic, and assumption-driven.
Assume 2-week sprints and one cross-functional product squad unless the user
states otherwise.

Always produce this structure:

# 1-Year Product Overview
## Product Vision
## Planning Assumptions
## Annual Goals
## Quarter 1
Include theme, outcomes, major capabilities, risks, and exit criteria.
## Quarter 2
Include theme, outcomes, major capabilities, risks, and exit criteria.
## Quarter 3
Include theme, outcomes, major capabilities, risks, and exit criteria.
## Quarter 4
Include theme, outcomes, major capabilities, risks, and exit criteria.

# Quarter 1 Delivery Plan
## Quarter 1 Objectives
## Epics
List the epics with business value, owner role, dependencies, and acceptance criteria.
## Sprint Plan
Plan 6 two-week sprints. For each sprint include sprint goal, epics touched,
user stories, acceptance criteria, dependencies, and demo outcome.
## Release Plan
## Metrics and Review Cadence
## Risks, Tradeoffs, and Open Questions
""".strip()


def log_uncaught_exception(exc_type, exc_value, exc_traceback) -> None:
    traceback.print_exception(exc_type, exc_value, exc_traceback)


@dataclass
class MLXChatEngine:
    model_name: str = DEFAULT_MODEL

    def __post_init__(self) -> None:
        self.model, self.tokenizer = load(self.model_name)

    def _prompt_for_idea(self, idea: str, system_prompt: str, request: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

    def stream_document(self, idea: str, mode: str):
        if mode == "plan":
            system_prompt = PLAN_SYSTEM_PROMPT
            request = (
                "Using the product brief below as the source of truth, create a "
                "1-year product overview split into 4 quarters, then plan epics "
                "and stories sprint-wise for Quarter 1.\n\n"
                "# Source Product Brief\n\n"
                f"{idea}"
            )
        else:
            system_prompt = BRIEF_SYSTEM_PROMPT
            request = f"Create the product brief and Amazon-style PRD for this idea:\n\n{idea}"

        prompt = self._prompt_for_idea(idea, system_prompt, request)
        try:
            from mlx_lm import stream_generate
        except ImportError:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                verbose=False,
            ).strip()
            yield response
            return

        for chunk in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
        ):
            text = getattr(chunk, "text", chunk)
            if text:
                yield str(text)


class ChatWorker(QObject):
    chunk = pyqtSignal(str)
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, engine: MLXChatEngine, message: str, mode: str) -> None:
        super().__init__()
        self.engine = engine
        self.message = message
        self.mode = mode

    def run(self) -> None:
        try:
            pending = []
            pending_chars = 0
            for text in self.engine.stream_document(self.message, self.mode):
                pending.append(text)
                pending_chars += len(text)
                if pending_chars >= 80 or "\n" in text:
                    self.chunk.emit("".join(pending))
                    pending.clear()
                    pending_chars = 0
            if pending:
                self.chunk.emit("".join(pending))
            self.finished.emit()
        except BaseException:  # pragma: no cover - keeps UI from freezing on model errors.
            self.failed.emit(traceback.format_exc())


class MessageBubble(QLabel):
    def __init__(self, text: str, role: str) -> None:
        super().__init__(text)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.setObjectName(f"{role}Bubble")


class ChatWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Productize")
        self.resize(900, 680)
        self.engine = MLXChatEngine()
        self.thread: QThread | None = None
        self.worker: ChatWorker | None = None
        self.streaming_response = ""
        self.current_mode = ""
        self.latest_brief = ""
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = QLabel("Productize")
        title.setObjectName("title")
        layout.addWidget(title)

        self.messages = QVBoxLayout()
        self.messages.setAlignment(Qt.AlignTop)
        self.messages.setSpacing(10)

        message_host = QWidget()
        message_host.setLayout(self.messages)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(message_host)
        self.scroll_area.setObjectName("chatSurface")
        layout.addWidget(self.scroll_area, 1)

        input_row = QHBoxLayout()
        input_row.setSpacing(12)

        self.input = QTextEdit()
        self.input.setPlaceholderText("Describe a product idea...")
        self.input.setFixedHeight(92)
        input_row.addWidget(self.input, 1)

        self.send_button = QPushButton("Create Brief")
        self.send_button.clicked.connect(self.create_brief)
        input_row.addWidget(self.send_button)

        self.plan_button = QPushButton("Plan Roadmap")
        self.plan_button.clicked.connect(self.plan_roadmap)
        input_row.addWidget(self.plan_button)

        layout.addLayout(input_row)
        self.setCentralWidget(root)

        self._add_message(
            "Describe an idea, create a brief, then plan a roadmap from that brief.",
            "assistant",
        )

    def create_brief(self) -> None:
        idea = self.input.toPlainText().strip()
        if not idea:
            return
        self.input.clear()
        self.latest_brief = ""
        self._start_generation(
            mode="brief",
            source_text=idea,
            loading_text="Creating brief...",
            user_message=idea,
        )

    def plan_roadmap(self) -> None:
        if not self.latest_brief:
            self._add_message(
                "Create a brief first, then I can plan the roadmap from it.",
                "assistant",
            )
            return
        self._start_generation(
            mode="plan",
            source_text=self.latest_brief,
            loading_text="Planning roadmap from the latest brief...",
        )

    def _start_generation(
        self,
        mode: str,
        source_text: str,
        loading_text: str,
        user_message: str | None = None,
    ) -> None:
        if not source_text or self.thread is not None:
            return

        self._set_busy(True)
        self.current_mode = mode
        self.streaming_response = ""
        if user_message:
            self._add_message(user_message, "user")
        self._add_message(loading_text, "assistant")

        self.thread = QThread()
        self.worker = ChatWorker(self.engine, source_text, mode)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.chunk.connect(self._handle_response_chunk)
        self.worker.finished.connect(self._handle_response_finished)
        self.worker.failed.connect(self._handle_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._cleanup_worker)
        self.thread.start()

    def _handle_response_chunk(self, chunk: str) -> None:
        try:
            self.streaming_response += chunk
            self._replace_last_assistant_message(self.streaming_response)
        except Exception:
            self._handle_error(traceback.format_exc())

    def _handle_response_finished(self) -> None:
        try:
            if self.streaming_response:
                final_response = self.streaming_response.strip()
                self._replace_last_assistant_message(final_response)
                if self.current_mode == "brief":
                    self.latest_brief = final_response
        except Exception:
            self._handle_error(traceback.format_exc())

    def _handle_error(self, error: str) -> None:
        self._replace_last_assistant_message(f"Error: {error}")

    def _cleanup_worker(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
        if self.thread is not None:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None
        self.current_mode = ""
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self.input.setEnabled(not busy)
        self.send_button.setEnabled(not busy)
        self.plan_button.setEnabled(not busy)

    def _add_message(self, text: str, role: str) -> None:
        row = QHBoxLayout()
        bubble = MessageBubble(text, role)
        row.addStretch(1 if role == "user" else 0)
        row.addWidget(bubble, 4)
        row.addStretch(0 if role == "user" else 1)
        self.messages.addLayout(row)
        self._scroll_to_bottom()

    def _replace_last_assistant_message(self, text: str) -> None:
        for index in range(self.messages.count() - 1, -1, -1):
            row = self.messages.itemAt(index).layout()
            if row is None:
                continue
            for child_index in range(row.count()):
                widget = row.itemAt(child_index).widget()
                if isinstance(widget, MessageBubble) and widget.objectName() == "assistantBubble":
                    widget.setText(text)
                    self._scroll_to_bottom()
                    return

    def _scroll_to_bottom(self) -> None:
        QTimer.singleShot(0, self._finish_scroll_to_bottom)

    def _finish_scroll_to_bottom(self) -> None:
        bar = self.scroll_area.verticalScrollBar()
        bar.setValue(bar.maximum())


def main() -> int:
    sys.excepthook = log_uncaught_exception
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_teal.xml")
    app.setStyleSheet(
        app.styleSheet()
        + """
        QLabel#title {
            font-size: 22px;
            font-weight: 600;
            padding-bottom: 4px;
        }
        QScrollArea#chatSurface {
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 8px;
        }
        QLabel#userBubble, QLabel#assistantBubble {
            border-radius: 8px;
            padding: 12px 14px;
            font-size: 14px;
            line-height: 1.35;
        }
        QLabel#userBubble {
            background-color: #00695c;
            color: #ffffff;
        }
        QLabel#assistantBubble {
            background-color: #263238;
            color: #ffffff;
        }
        """
    )
    window = ChatWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
