# --- Python 3.10+ 호환성 패치 ---
import collections, collections.abc
for name in ["Hashable","Iterable","Iterator","Mapping","MutableMapping"]:
    if not hasattr(collections, name):
        setattr(collections, name, getattr(collections.abc, name))

# FastAPI
from fastapi import FastAPI
from pydantic import BaseModel

# 내부 로직 import
from interface.nl2kqml import nl_to_kqml
from interface.kqml2nl import kqml_to_nl
from belief.knowledge_base import build_kb
from belief.state_manager import StateManager
from desire.goal_manager import GoalManager
from desire.curriculum import Curriculum
from desire.progress_tracker import ProgressTracker
from intention.planner import Planner
from intention.executor import Executor
from intention.feedback_agent import FeedbackAgent

# FastAPI 앱 객체
app = FastAPI()

# ==== 공통 로직 함수 ====
def run_bdi_pipeline(question: str):
    state = StateManager()
    eng = build_kb()

    atom = nl_to_kqml(question)
    results = eng.query(atom)
    if not results:
        return {"error": "Belief 결과 없음"}

    answer = kqml_to_nl(atom.substitute(results[0]))

    # Desire 단계
    goal_manager = GoalManager()
    curriculum = Curriculum()
    progress = ProgressTracker(state)
    curriculum.add("short_term", question)
    goal_manager.add(answer, priority=2)

    # Intention 단계
    planner = Planner()
    executor = Executor()
    feedback = FeedbackAgent()

    goal = goal_manager.get_highest_priority()
    plan = planner.make_plan(goal["goal"])
    exec_results = executor.execute(plan)
    fb = feedback.evaluate(exec_results)

    return {
        "NL→KQML": str(atom),
        "KQML→NL": answer,
        "Desire": goal_manager.show(),
        "Plan": plan,
        "Execution": exec_results,
        "Feedback": fb,
    }

# ==== FastAPI 엔드포인트 ====
class QueryInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "BDI Chat UI is running!"}

@app.post("/ask")
def ask_question(data: QueryInput):
    return run_bdi_pipeline(data.text)

# ==== 기존 main 함수 유지 ====
def main():
    print("=== BDI Tutor Chat ===\n")
    question = input("[You] ")
    results = run_bdi_pipeline(question)

    # CLI 출력
    print("[NL → KQML]", results.get("NL→KQML"))
    print("[KQML → NL]", results.get("KQML→NL"), "\n")
    print("[Desire: 목표]", results.get("Desire"))

    print("\n[Intention: 계획 수립]")
    for step in results.get("Plan", []):
        print("-", step)

    print("\n[Intention: 실행 결과]")
    for step in results.get("Execution", []):
        print("-", step)

    print("\n[Intention: 피드백]")
    print(results.get("Feedback"))
    
    
    
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# 정적 파일 제공 (CSS/JS 있으면 여기 넣음)

@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <title>BDI Chat</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            #chat-box { border: 1px solid #ccc; padding: 10px; height: 400px; overflow-y: auto; }
            .user { color: blue; margin: 5px 0; }
            .bot { color: green; margin: 5px 0; }
        </style>
    </head>
    <body>
        <h2>BDI Chat Interface</h2>
        <div id="chat-box"></div>
        <input type="text" id="user-input" placeholder="질문 입력..." style="width:80%"/>
        <button onclick="sendMessage()">전송</button>

        <script>
            async function sendMessage() {
                const input = document.getElementById("user-input");
                const message = input.value;
                if (!message) return;

                // 사용자 메시지 표시
                const chatBox = document.getElementById("chat-box");
                chatBox.innerHTML += "<div class='user'>You: " + message + "</div>";

                // 서버에 전송
                const res = await fetch("/ask", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: message })
                });
                const data = await res.json();

                // 봇 응답 표시 (KQML→NL만 보여주고 싶을 때)
                chatBox.innerHTML += "<div class='bot'>Bot: " + (data["KQML→NL"] || JSON.stringify(data)) + "</div>";
                chatBox.scrollTop = chatBox.scrollHeight;

                input.value = "";
            }
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    main()
