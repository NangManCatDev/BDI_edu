# Note: Gradio 기반 실행 진입점
# Fixme: 현재 이 코드는 KnowOrNot 리포지토리에서 가져온 것으로, 이 프로젝트에 맞춰서 수정할 것.

"""
run.py
------
BDI Tutor 프레임워크 실행 진입점 (Gradio UI 기반)
"""

import gradio as gr
from orchestrator.belief_manager import BeliefManager
from orchestrator.intention_selector import IntentionSelector
from orchestrator.plan_executor import PlanExecutor



# --- 모듈 초기화 ---
belief_manager = BeliefManager()
desire_manager = DesireManager()
intention_selector = IntentionSelector()
plan_executor = PlanExecutor()


def tutor_chat(user_input, history=[]):
    """
    Gradio 채팅 핸들러
    - user_input: 학생 입력(질문, 답변 등)
    - history: 기존 대화 기록
    """

    # 1. Belief 업데이트
    belief_state = belief_manager.update(user_input)

    # 2. Desire 설정 (더미: 세션 시작 시 자동 세팅)
    if not desire_manager.get_goal():
        desire_manager.set_goal("일제 강점기 경제 수탈 정책 이해하기")

    # 3. Intention 선택
    selected_plan = intention_selector.select(belief_state)

    # 4. Plan 실행
    response = plan_executor.execute(selected_plan, user_input)

    history.append((user_input, response))
    return history, history


# --- Gradio UI 정의 ---
with gr.Blocks() as demo:
    gr.Markdown("# 📘 BDI Tutor Prototype (한국사 예시)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="학생 입력")
    clear = gr.Button("대화 초기화")

    msg.submit(tutor_chat, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: ([], []), None, [chatbot, chatbot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
