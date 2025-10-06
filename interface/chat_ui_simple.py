# --- Python 3.10+ 호환성 패치 ---
import collections, collections.abc
for name in ["Hashable","Iterable","Iterator","Mapping","MutableMapping"]:
    if not hasattr(collections, name):
        setattr(collections, name, getattr(collections.abc, name))

# FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import logging
from typing import Dict, Any, Optional, List

# 내부 로직 import
from .bdi_flow_components import BDIFlowOrchestrator
from belief.state_manager import StateManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 상태 관리
user_sessions: Dict[str, StateManager] = {}
conversation_history: Dict[str, List[Dict[str, str]]] = {}

# FastAPI 앱 객체
app = FastAPI(title="BDI Tutor Chat", version="1.0.0")

# ==== 헬퍼 함수 ====
def get_user_state(user_id: str = "default") -> StateManager:
    """사용자별 상태 관리자 반환"""
    if user_id not in user_sessions:
        user_sessions[user_id] = StateManager()
        conversation_history[user_id] = []
        logger.info(f"새 사용자 세션 생성: {user_id}")
    return user_sessions[user_id]

def get_conversation_context(user_id: str, max_history: int = 3) -> str:
    """사용자의 대화 이력을 컨텍스트로 반환"""
    if user_id not in conversation_history:
        return ""
    
    history = conversation_history[user_id][-max_history:]
    context_parts = []
    
    for entry in history:
        if entry.get("question"):
            context_parts.append(f"이전 질문: {entry['question']}")
        if entry.get("answer"):
            context_parts.append(f"이전 답변: {entry['answer'][:100]}...")
    
    return "\n".join(context_parts) if context_parts else ""

def add_to_conversation_history(user_id: str, question: str, answer: str):
    """대화 이력에 추가"""
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    
    conversation_history[user_id].append({
        "question": question,
        "answer": answer,
        "timestamp": "2024-01-01T00:00:00Z"  # 실제로는 datetime.now().isoformat()
    })
    
    # 최대 10개까지만 유지
    if len(conversation_history[user_id]) > 10:
        conversation_history[user_id] = conversation_history[user_id][-10:]

def _substitute_atom(atom, bindings):
    """Atom의 변수를 바인딩 값으로 대체"""
    new_args = []
    for arg in atom.arguments:
        if hasattr(arg, 'name'):  # Variable인 경우
            var_name = arg.name
            if var_name in bindings:
                new_args.append(bindings[var_name])
            else:
                new_args.append(arg)
        else:
            new_args.append(arg)
    
    # 새로운 Atom 객체 생성
    return type('Atom', (), {
        'predicate': atom.predicate,
        'arguments': new_args
    })()

def analyze_question_type(question: str) -> str:
    """질문 유형 분석"""
    question_lower = question.lower()
    if any(word in question_lower for word in ["언제", "몇 년", "년도"]):
        return "temporal"
    elif any(word in question_lower for word in ["누구", "누가", "인물"]):
        return "person"
    elif any(word in question_lower for word in ["왜", "원인", "이유"]):
        return "causal"
    elif any(word in question_lower for word in ["어떻게", "방법", "과정"]):
        return "process"
    else:
        return "general"

# ==== BDI 플로우 오케스트레이터 ====
bdi_orchestrator = BDIFlowOrchestrator()

def run_bdi_pipeline(question: str, user_id: str = "default") -> Dict[str, Any]:
    """BDI 플로우차트 기반 파이프라인 실행"""
    try:
        logger.info(f"사용자 {user_id} 입력: {question}")
        
        # 입력 검증
        if not question or len(question.strip()) < 2:
            return {"error": "질문이 너무 짧습니다. 2글자 이상 입력해주세요."}
        
        if len(question) > 1000:
            return {"error": "질문이 너무 깁니다. 1000글자 이하로 입력해주세요."}
        
        # BDI 플로우 실행
        result = bdi_orchestrator.process(question, user_id)
        
        if result["success"]:
            # 대화 이력에 추가
            add_to_conversation_history(user_id, question, result["final_answer"])
            logger.info(f"사용자 {user_id} 응답 생성 완료")
        
        return result
        
    except Exception as e:
        logger.error(f"BDI 파이프라인 전체 오류: {str(e)}")
        return {"error": f"시스템 오류가 발생했습니다: {str(e)}"}

# ==== FastAPI 엔드포인트 ====
class QueryInput(BaseModel):
    text: str
    user_id: Optional[str] = "default"
    
    @validator('text')
    def validate_text(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('질문은 2글자 이상이어야 합니다.')
        if len(v) > 1000:
            raise ValueError('질문은 1000글자 이하여야 합니다.')
        return v.strip()

class QueryResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

@app.get("/")
def root():
    return {
        "message": "BDI Tutor Chat API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/chat",
            "ask": "/ask (POST)",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "active_sessions": len(user_sessions),
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.post("/ask", response_model=QueryResponse)
def ask_question(data: QueryInput):
    """사용자 질문을 처리하고 BDI 파이프라인을 실행합니다."""
    try:
        result = run_bdi_pipeline(data.text, data.user_id)
        
        if "error" in result:
            return QueryResponse(
                success=False,
                error=result["error"]
            )
        else:
            return QueryResponse(
                success=True,
                message="질문이 성공적으로 처리되었습니다.",
                data=result
            )
            
    except Exception as e:
        logger.error(f"API 엔드포인트 오류: {str(e)}")
        return QueryResponse(
            success=False,
            error=f"서버 내부 오류가 발생했습니다: {str(e)}"
        )

# ==== CLI 인터페이스 ====
def main():
    print("🎓 BDI Tutor Chat - CLI 모드")
    print("=" * 50)
    print("한국사 AI 튜터와 대화해보세요!")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
    
    user_id = "cli_user"
    
    while True:
        try:
            question = input("💬 [You] ").strip()
            
            if question.lower() in ['quit', 'exit', '종료']:
                print("👋 안녕히 가세요!")
                break
                
            if not question:
                print("❌ 질문을 입력해주세요.")
                continue
                
            print("🤔 AI가 답변을 생성하고 있습니다...")
            results = run_bdi_pipeline(question, user_id)
            
            if "error" in results:
                print(f"❌ 오류: {results['error']}")
                continue
                
            # 성공 응답 출력
            print("\n" + "="*50)
            print("📝 [답변]")
            print(f"🤖 {results.get('final_answer', '답변을 생성할 수 없습니다.')}")
            
            print("\n🔍 [처리 과정]")
            print(f"📊 의도: {results.get('intention', 'N/A')}")
            print(f"📊 정제된 입력: {results.get('refined_input', 'N/A')}")
            
            result_data = results.get('result', {})
            if result_data.get('type') == 'answer':
                print(f"📊 쿼리: {result_data.get('query', 'N/A')}")
            elif result_data.get('type') == 'desire':
                print(f"📊 목표: {result_data.get('goal', 'N/A')}")
            elif result_data.get('type') == 'task':
                print(f"📊 작업: {result_data.get('task', 'N/A')}")
                print(f"📊 계획: {result_data.get('plan', [])}")
            
            print("="*50 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 안녕히 가세요!")
            break
        except Exception as e:
            print(f"❌ 예상치 못한 오류가 발생했습니다: {str(e)}")
            continue

from fastapi.responses import HTMLResponse

@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <title>BDI Tutor Chat</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                overflow: hidden;
                display: flex;
                min-height: 600px;
            }
            .main-content {
                flex: 1;
                display: flex;
                flex-direction: column;
            }
            .debug-sidebar {
                width: 350px;
                background: #f8f9fa;
                border-left: 2px solid #dee2e6;
                display: flex;
                flex-direction: column;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                text-align: center;
            }
            .header h1 {
                margin: 0;
                font-size: 24px;
            }
            .header p {
                margin: 5px 0 0 0;
                opacity: 0.9;
            }
            #chat-box { 
                border: none;
                padding: 20px; 
                height: 400px; 
                overflow-y: auto; 
                background: #fafafa;
                flex: 1;
            }
            .message {
                margin: 10px 0;
                padding: 10px 15px;
                border-radius: 15px;
                max-width: 80%;
                word-wrap: break-word;
            }
            .user { 
                background: #007bff;
                color: white;
                margin-left: auto;
                text-align: right;
            }
            .bot { 
                background: #e9ecef;
                color: #333;
                margin-right: auto;
            }
            .error {
                background: #dc3545;
                color: white;
            }
            .input-area {
                padding: 20px;
                border-top: 1px solid #eee;
                display: flex;
                gap: 10px;
            }
            #user-input {
                flex: 1;
                padding: 12px 15px;
                border: 1px solid #ddd;
                border-radius: 25px;
                outline: none;
                font-size: 14px;
            }
            #user-input:focus {
                border-color: #007bff;
                box-shadow: 0 0 0 2px rgba(0,123,255,0.25);
            }
            .send-btn {
                padding: 12px 25px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-weight: bold;
                transition: background 0.3s;
            }
            .send-btn:hover {
                background: #0056b3;
            }
            .send-btn:disabled {
                background: #6c757d;
                cursor: not-allowed;
            }
            .loading {
                display: none;
                text-align: center;
                color: #666;
                font-style: italic;
            }
            .debug-sidebar-header {
                background: #495057;
                color: white;
                padding: 10px;
                text-align: center;
                font-weight: bold;
            }
            .debug-info {
                display: grid;
                grid-template-columns: 1fr;
                gap: 8px;
                margin: 10px;
            }
            .debug-item {
                background: #e9ecef;
                padding: 6px;
                border-radius: 3px;
                border-left: 3px solid #007bff;
                font-size: 11px;
            }
            .debug-item strong {
                color: #495057;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- 좌측 디버깅 사이드바 -->
            <div class="debug-sidebar">
                <div class="debug-sidebar-header">
                    🔍 BDI 디버그 로그
                </div>
                
                <div class="debug-info" id="debug-info">
                    <div class="debug-item">
                        <strong>시스템 상태:</strong> <span id="system-status">대기 중</span>
                    </div>
                    <div class="debug-item">
                        <strong>처리 시간:</strong> <span id="processing-time">-</span>
                    </div>
                    <div class="debug-item">
                        <strong>의도 분류:</strong> <span id="intention-type">-</span>
                    </div>
                    <div class="debug-item">
                        <strong>KB 상태:</strong> <span id="kb-status">-</span>
                    </div>
                </div>
            </div>
            
            <div class="main-content">
                <div class="header">
                    <h1>🎓 BDI Tutor Chat</h1>
                    <p>한국사 AI 튜터와 대화해보세요</p>
                </div>
                
                <div id="chat-box">
                    <div class="message bot">
                        안녕하세요! 한국사 AI 튜터입니다. 궁금한 한국사 질문을 해주세요.
                    </div>
                </div>
                
                <div class="loading" id="loading">AI가 답변을 생성하고 있습니다...</div>
                
                <div class="input-area">
                    <input type="text" id="user-input" placeholder="한국사 질문을 입력하세요..." />
                    <button class="send-btn" id="send-btn" onclick="sendMessage()">전송</button>
                </div>
            </div>
        </div>

        <script>
            function sendMessage() {
                console.log("sendMessage 함수 호출됨");
                
                const userInput = document.getElementById("user-input");
                if (!userInput) {
                    console.error("userInput 요소를 찾을 수 없습니다!");
                    return;
                }
                
                const message = userInput.value.trim();
                console.log("입력된 메시지:", message);
                
                if (!message) {
                    console.log("메시지가 비어있습니다.");
                    return;
                }
                
                console.log("sendMessage 처리 시작:", message);
                
                // 비동기 처리
                handleMessage(message);
            }
            
            async function handleMessage(message) {
                console.log("handleMessage 시작:", message);
                
                // 사용자 메시지 표시
                addMessage(message, "user");
                const userInput = document.getElementById("user-input");
                if (userInput) {
                    userInput.value = "";
                }
                
                try {
                    // 서버에 전송
                    const response = await fetch("/ask", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ 
                            text: message,
                            user_id: "web_user"
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        // 성공 응답 표시
                        const botMessage = data.data["final_answer"] || "답변을 생성했습니다.";
                        addMessage(botMessage, "bot");
                    } else {
                        addMessage("오류: " + data.error, "error");
                    }
                    
                } catch (error) {
                    addMessage("네트워크 오류가 발생했습니다.", "error");
                }
            }
            
            function addMessage(text, type) {
                const chatBox = document.getElementById("chat-box");
                if (!chatBox) return;
                
                const messageDiv = document.createElement("div");
                messageDiv.className = `message ${type}`;
                messageDiv.textContent = text;
                chatBox.appendChild(messageDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            }
            
            // Enter 키로 전송
            document.addEventListener('DOMContentLoaded', function() {
                const userInput = document.getElementById("user-input");
                if (userInput) {
                    userInput.addEventListener("keypress", function(e) {
                        if (e.key === "Enter") {
                            sendMessage();
                        }
                    });
                }
            });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    main()
