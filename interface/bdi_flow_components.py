#!/usr/bin/env python3
"""
BDI 시스템 플로우차트 컴포넌트들
플로우차트에 맞춘 각 모듈 구현
"""

import logging
from typing import Dict, Any, Optional, List
from .nl2kqml import nl_to_kqml
from .kqml2nl import kqml_to_nl
from belief.knowledge_base import build_kb
from belief.state_manager import StateManager
from desire.goal_manager import GoalManager
from desire.curriculum import Curriculum
from desire.progress_tracker import ProgressTracker
from intention.planner import Planner
from intention.executor import Executor
from intention.feedback_agent import FeedbackAgent

logger = logging.getLogger(__name__)

class InputRefiner:
    """입력 메시지 정제"""
    def refine(self, message: str) -> str:
        """사용자 입력을 정제합니다"""
        # 기본적인 입력 정제
        refined = message.strip()
        
        # 특수문자 정리
        refined = refined.replace('\n', ' ').replace('\t', ' ')
        
        # 연속 공백 제거
        while '  ' in refined:
            refined = refined.replace('  ', ' ')
        
        logger.info(f"입력 정제: '{message}' → '{refined}'")
        return refined

class CoreferenceResolver:
    """간단 지시어(그/그녀/그 시기 등) 해소기"""
    def __init__(self):
        self.entities = [
            "이순신", "세종", "세종대왕", "이성계", "정도전", "신사임당", "이황", "이이",
            "김구", "안중근", "윤봉길",
            "임진왜란", "동학농민운동", "3.1운동", "광복절", "한글창제", "훈민정음반포", "조선 건국"
        ]
        self.pronouns = ["그에 대한", "그에 관해", "그에 대해", "그는", "그녀는", "그", "그녀", "그 사람",
                          "그 시기", "그때", "그 당시", "그 사건", "그 일"]

    def _pick_focus_from_context(self, context: str) -> str | None:
        if not context:
            return None
        # 가장 최근 등장하는 엔티티를 선택
        last_pos = -1
        last_ent = None
        for ent in self.entities:
            pos = context.rfind(ent)
            if pos > last_pos:
                last_pos = pos
                last_ent = ent
        return last_ent

    def resolve(self, message: str, context: str | None) -> str:
        if not message:
            return message
        if not context:
            return message
        # 지시어가 포함된 경우에만 치환
        if not any(p in message for p in self.pronouns):
            return message
        focus = self._pick_focus_from_context(context)
        if not focus:
            return message
        resolved = message
        # 주제 지시어를 focus로 치환
        replacements = {
            "그에 대한": f"{focus}에 대한",
            "그에 관해": f"{focus}에 관해",
            "그에 대해": f"{focus}에 대해",
            "그 사람": focus,
            "그녀는": focus,
            "그는": focus,
            "그녀": focus,
            "그": focus,
            "그 시기": f"{focus} 시기",
            "그때": f"{focus} 당시",
            "그 당시": f"{focus} 당시",
            "그 사건": focus,
            "그 일": focus
        }
        for k, v in replacements.items():
            resolved = resolved.replace(k, v)
        logger.info(f"지시어 해소: '{message}' -> '{resolved}' (focus={focus})")
        return resolved

class IntentionClassifier:
    """상황 정보 기반 메시지 의도 분석"""
    
    def __init__(self):
        self.question_keywords = ["누구", "무엇", "언제", "어디", "왜", "어떻게", "?", "알려줘", "설명해줘"]
        self.desire_keywords = ["원해", "하고싶", "알고싶", "배우고싶", "학습하고싶", "공부하고싶"]
        self.task_keywords = ["해줘", "실행", "시작", "계속", "다음", "진행"]
        # 세부 의도 키워드
        self.quiz_keywords = ["문제", "퀴즈", "문항", "내줘"]
        self.summarize_keywords = ["요약", "정리"]
        self.compare_keywords = ["비교", "차이"]
        self.why_keywords = ["왜", "원인", "이유"]
        self.how_keywords = ["어떻게", "과정", "절차"]
        self.when_keywords = ["언제", "연도", "년도", "연대"]
        self.who_keywords = ["누구", "인물"]
        # Function/StaticText 트리거
        self.function_keywords = ["시간", "현재시각", "서버상태"]
        self.static_keywords = ["도움말", "사용법", "help"]
    
    def classify(self, message: str) -> str:
        """메시지 의도를 분류합니다"""
        message_lower = message.lower()
        if len(message_lower) < 2:
            logger.info("의도 분류: incorrect - 입력이 너무 짧음")
            return "incorrect"
        
        # 질문 패턴 감지
        if any(keyword in message_lower for keyword in self.question_keywords):
            logger.info(f"의도 분류: Question - {message}")
            return "question"
        
        # 욕구 패턴 감지
        if any(keyword in message_lower for keyword in self.desire_keywords):
            logger.info(f"의도 분류: Desire - {message}")
            return "desire"
        
        # 작업 패턴 감지
        if any(keyword in message_lower for keyword in self.task_keywords):
            logger.info(f"의도 분류: Task - {message}")
            return "task"
        # 세부 의도
        if any(k in message_lower for k in self.quiz_keywords):
            return "quiz"
        if any(k in message_lower for k in self.summarize_keywords):
            return "summarize"
        if any(k in message_lower for k in self.compare_keywords):
            return "compare"
        if any(k in message_lower for k in self.why_keywords):
            return "why"
        if any(k in message_lower for k in self.how_keywords):
            return "how"
        if any(k in message_lower for k in self.when_keywords):
            return "when"
        if any(k in message_lower for k in self.who_keywords):
            return "who"
        if any(k in message_lower for k in self.function_keywords):
            return "function"
        if any(k in message_lower for k in self.static_keywords):
            return "static_text"
        
        # 기본값은 질문으로 처리
        logger.info(f"의도 분류: Question (default) - {message}")
        return "question"

class QAAgent:
    """고객 Goal 분석 및 분류 - 질문 처리"""
    
    def __init__(self):
        self.kb_engine = None
        self._init_kb()
    
    def _init_kb(self):
        """지식베이스 초기화 - 이미 로드된 NEO 엔진 공유 우선"""
        try:
            try:
                from .chat_ui import neo_kb_loader
                if neo_kb_loader:
                    self.kb_engine = neo_kb_loader.get_engine()
                    logger.info("QA Agent: 공유된 NEO 엔진 사용")
                    return
            except Exception:
                pass
            self.kb_engine = build_kb()
            logger.info("QA Agent 지식베이스 초기화 완료")
        except Exception as e:
            logger.error(f"QA Agent 지식베이스 초기화 실패: {str(e)}")
            self.kb_engine = None
    
    def process(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        """질문을 처리합니다"""
        logger.info(f"QA Agent 처리: {question}")
        
        if not self.kb_engine:
            return {"type": "error", "message": "지식베이스를 초기화할 수 없습니다."}
        
        try:
            # NL → KQML 변환
            atom = nl_to_kqml(question)
            if atom is None:
                return {"type": "error", "message": "질문을 이해할 수 없습니다."}
            
            # KB 쿼리 실행
            results = self.kb_engine.query(atom)
            if not results or len(results) == 0:
                # 폴백: NEO RAG로 쿼리 생성 후 동일 엔진에 문자열 쿼리 실행
                try:
                    from .chat_ui import neo_rag, neo_kb_loader
                    if neo_rag and neo_kb_loader:
                        conv = neo_rag.convert_to_neo_query(question)
                        if conv and conv.get("success"):
                            q = conv["query"]
                            exec_res = neo_kb_loader.get_engine().query_simple(q)
                            if exec_res:
                                nl = neo_rag.convert_neo_result_to_nl({"success": True, "results": exec_res, "query": q})
                                if nl and nl != "관련 정보를 찾을 수 없습니다.":
                                    return {"type": "answer", "content": nl, "query": q, "results": exec_res}
                except Exception as fe:
                    logger.warning(f"NEO RAG 폴백 실패: {str(fe)}")
                return {"type": "error", "message": "해당 질문에 대한 답변을 찾을 수 없습니다."}
            
            # KQML → NL 변환
            from .chat_ui import _substitute_atom
            substituted_atom = _substitute_atom(atom, results[0])
            # 대화 컨텍스트가 있으면 LLM 변환에 전달
            answer = kqml_to_nl(substituted_atom, context=context)
            
            return {
                "type": "answer",
                "content": answer,
                "query": str(atom),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"QA Agent 처리 실패: {str(e)}")
            return {"type": "error", "message": f"질문 처리 중 오류가 발생했습니다: {str(e)}"}

class DesireClassifier:
    """고객 Goal 분석 및 분류 - 욕구 분석"""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.goal_manager = GoalManager()
        self.curriculum = Curriculum()
        self.progress_tracker = ProgressTracker(self.state_manager)
    
    def process(self, desire: str) -> Dict[str, Any]:
        """욕구를 분석하고 목표를 설정합니다"""
        logger.info(f"Desire Classifier 처리: {desire}")
        
        try:
            # 욕구 분석 및 목표 설정
            goal = f"{desire}에 대한 학습 목표"
            self.curriculum.add("short_term", goal)
            self.goal_manager.add(goal, priority=2)
            
            # 진행 상황 업데이트
            self.progress_tracker.update_progress(goal, 0.1)
            
            return {
                "type": "desire",
                "goal": goal,
                "goals": self.goal_manager.show(),
                "curriculum": self.curriculum.show()
            }
            
        except Exception as e:
            logger.error(f"Desire Classifier 처리 실패: {str(e)}")
            return {"type": "error", "message": f"욕구 분석 중 오류가 발생했습니다: {str(e)}"}

class TaskManager:
    """고객 Goal 분석 및 분류 - 작업 관리"""
    
    def __init__(self):
        self.planner = Planner()
        self.executor = Executor()
        self.feedback = FeedbackAgent()
    
    def process(self, task: str) -> Dict[str, Any]:
        """작업을 관리하고 실행합니다"""
        logger.info(f"Task Manager 처리: {task}")
        
        try:
            # 작업 계획 수립
            plan = self.planner.make_plan(task)
            logger.info(f"작업 계획 수립: {len(plan)}개 단계")
            
            # 계획 실행
            exec_results = self.executor.execute(plan)
            logger.info(f"작업 실행 완료: {len(exec_results)}개 결과")
            
            # 피드백 생성
            feedback_result = self.feedback.evaluate(exec_results)
            logger.info("피드백 생성 완료")
            
            return {
                "type": "task",
                "task": task,
                "plan": plan,
                "execution": exec_results,
                "feedback": feedback_result
            }
            
        except Exception as e:
            logger.error(f"Task Manager 처리 실패: {str(e)}")
            return {"type": "error", "message": f"작업 처리 중 오류가 발생했습니다: {str(e)}"}

class BeliefManager:
    """Belief 조회 및 업데이트"""
    
    def __init__(self):
        self.kb_engine = None
        self.state_manager = StateManager()
        self._init_kb()
    
    def _init_kb(self):
        """지식베이스 초기화"""
        try:
            self.kb_engine = build_kb()
            logger.info("Belief Manager 지식베이스 초기화 완료")
        except Exception as e:
            logger.error(f"Belief Manager 지식베이스 초기화 실패: {str(e)}")
            self.kb_engine = None
    
    def query(self, query) -> List[Dict[str, Any]]:
        """KB에서 쿼리를 실행합니다"""
        if not self.kb_engine:
            logger.error("지식베이스가 초기화되지 않았습니다.")
            return []
        
        try:
            results = self.kb_engine.query(query)
            logger.info(f"Belief Manager 쿼리 실행: {len(results)}개 결과")
            return results
        except Exception as e:
            logger.error(f"Belief Manager 쿼리 실행 실패: {str(e)}")
            return []
    
    def update(self, new_info: str):
        """새로운 정보로 상태를 업데이트합니다"""
        try:
            self.state_manager.set_last_topic(new_info)
            logger.info(f"Belief Manager 상태 업데이트: {new_info}")
        except Exception as e:
            logger.error(f"Belief Manager 상태 업데이트 실패: {str(e)}")

class OutputRefiner:
    """출력 텍스트 정제"""
    
    def refine(self, output: Dict[str, Any]) -> str:
        """처리 결과를 자연스러운 텍스트로 정제합니다"""
        logger.info(f"Output Refiner 처리: {output.get('type', 'unknown')}")
        
        if output["type"] == "answer":
            return output["content"]
        
        elif output["type"] == "desire":
            goal = output.get("goal", "목표")
            return f"학습 목표가 설정되었습니다: {goal}"
        
        elif output["type"] == "task":
            task = output.get("task", "작업")
            feedback = output.get("feedback", "완료")
            return f"작업 '{task}'이 완료되었습니다. 피드백: {feedback}"
        
        elif output["type"] == "error":
            return f"오류가 발생했습니다: {output.get('message', '알 수 없는 오류')}"
        
        elif output["type"] == "quiz":
            return output.get("content", "문제를 생성하지 못했습니다.")
        
        elif output["type"] == "static_text":
            return output.get("content", "")
        
        elif output["type"] == "function_result":
            return output.get("content", "함수 결과가 없습니다.")
        
        elif output["type"] == "incorrect":
            return output.get("message", "질문을 이해하지 못했습니다. 더 구체적으로 다시 입력해 주세요.")
        
        else:
            return f"처리 결과: {str(output)}"

class FunctionCaller:
    """지정된 함수 호출"""
    
    def __init__(self):
        pass
    
    def call(self, refined_input: str) -> Dict[str, Any]:
        try:
            if "시간" in refined_input or "현재시각" in refined_input:
                import datetime
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return {"type": "function_result", "name": "current_time", "content": f"현재 시각은 {now} 입니다."}
            if "서버상태" in refined_input:
                return {"type": "function_result", "name": "server_status", "content": "서버는 정상 동작 중입니다."}
            return {"type": "function_result", "name": "noop", "content": "호출할 함수가 없습니다."}
        except Exception as e:
            return {"type": "error", "message": f"함수 호출 실패: {str(e)}"}

class StaticTextResponder:
    """지정된 텍스트 출력"""
    
    def __init__(self):
        self.snippets = {
            "도움말": "질문을 입력하면 한국사 지식베이스와 LLM을 통해 답변합니다. '문제 내줘', '요약', '비교', '원인', '과정' 등도 가능합니다.",
            "사용법": "예시) '이순신은 누구야', '임진왜란 원인 알려줘', '세종과 태종 비교', '세종대왕 요약'",
            "help": "Available intents: question, quiz, summarize, compare, why, how, who, when, function, static_text"
        }
    
    def respond(self, refined_input: str) -> Dict[str, Any]:
        for key, txt in self.snippets.items():
            if key in refined_input:
                return {"type": "static_text", "content": txt}
        return {"type": "static_text", "content": self.snippets["도움말"]}

class QuizGenerator:
    """LLM 기반 퀴즈 생성기 (동적으로 객관식 퀴즈 생성)"""
    
    def __init__(self, qa_agent: QAAgent):
        self.qa_agent = qa_agent
        self.known_entities = [
            "이순신", "세종", "세종대왕", "임진왜란", "이성계", "정도전",
            "김구", "안중근", "윤봉길", "동학농민운동", "3.1운동", "광복절"
        ]
        from .kqml2nl import get_connector
        self.llm = get_connector()
    
    def _to_info_prompt(self, refined_input: str) -> str:
        # "문제", "퀴즈", "내줘" 등을 정보 요청 형태로 치환
        tmp = refined_input
        # 먼저 "알려줘"가 이미 있는지 확인
        if "알려줘" in tmp:
            # 이미 "알려줘"가 있으면 그대로 사용
            return tmp
        # "알려줘"가 없으면 치환
        for k in ["문제", "퀴즈", "문항", "내줘", "내 줘", "출제", "내려줘"]:
            tmp = tmp.replace(k, "알려줘")
        # 중복 표현 정리
        while "알려줘알려줘" in tmp:
            tmp = tmp.replace("알려줘알려줘", "알려줘")
        tmp = tmp.replace("알려줘를 알려줘", "알려줘")
        tmp = tmp.replace("알려줘 를 알려줘", "알려줘")
        return tmp
    
    def _generate_quiz_with_llm(self, topic: str, context_info: str) -> Dict[str, Any]:
        """LLM을 사용하여 동적으로 퀴즈 생성"""
        try:
            system_prompt = """당신은 한국사 교육 전문가입니다. 주어진 주제에 대해 고등학교 수준의 객관식 퀴즈를 생성해주세요.

퀴즈 생성 규칙:
1. 문제는 명확하고 구체적으로 작성
2. 5개의 선택지를 제공 (정답 1개 + 오답 4개)
3. 오답은 그럴듯하지만 틀린 내용으로 구성
4. 정답은 정확한 역사적 사실에 기반
5. 한국어로 작성

출력 형식:
문제: [질문 내용]
① [선택지 1]
② [선택지 2]  
③ [선택지 3]
④ [선택지 4]
⑤ [선택지 5]
정답: [정답 번호] - [정답 설명]"""

            user_prompt = f"""주제: {topic}

관련 정보:
{context_info}

위 정보를 바탕으로 객관식 퀴즈를 생성해주세요."""

            response = self.llm.ask(user_prompt, system_prompt)
            
            # 응답 파싱
            lines = response.strip().split('\n')
            question = ""
            options = []
            answer = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith('문제:'):
                    question = line.replace('문제:', '').strip()
                elif line.startswith('①') or line.startswith('②') or line.startswith('③') or line.startswith('④') or line.startswith('⑤'):
                    options.append(line)
                elif line.startswith('정답:'):
                    answer = line.replace('정답:', '').strip()
            
            if question and options and answer:
                return {
                    "success": True,
                    "question": question,
                    "options": options,
                    "answer": answer
                }
            else:
                return {"success": False, "error": "퀴즈 파싱 실패"}
                
        except Exception as e:
            logger.error(f"LLM 퀴즈 생성 실패: {str(e)}")
            return {"success": False, "error": str(e)}

    def process(self, refined_input: str, context: Optional[str] = None) -> Dict[str, Any]:
        try:
            # 컨텍스트에서 핵심 엔티티 추출
            focus = None
            if context:
                for ent in self.known_entities:
                    if ent in context:
                        focus = ent
                        break
            
            info_prompt = self._to_info_prompt(refined_input)
            # 엔티티가 있으면 정보 요청을 해당 엔티티로 보정
            if focus and focus not in info_prompt:
                info_prompt = f"{focus}에 대해 알려줘"
            
            # 관련 정보 수집
            qa = self.qa_agent.process(info_prompt, context=context)
            context_info = ""
            if qa.get("type") == "answer" and qa.get("content"):
                context_info = qa["content"].strip()
            
            # 주제 결정
            if focus:
                topic = focus
            elif "이순신" in refined_input:
                topic = "이순신"
            elif "세종" in refined_input or "세종대왕" in refined_input:
                topic = "세종대왕"
            elif "임진왜란" in refined_input:
                topic = "임진왜란"
            else:
                topic = refined_input.replace("문제", "").replace("퀴즈", "").replace("내줘", "").strip()
            
            # LLM으로 퀴즈 생성
            quiz_result = self._generate_quiz_with_llm(topic, context_info)
            
            if quiz_result.get("success"):
                # 퀴즈 내용을 문자열로 포맷팅
                content = f"문제: {quiz_result['question']}\n"
                for option in quiz_result['options']:
                    content += f"{option}\n"
                content += f"정답: {quiz_result['answer']}"
                
                return {
                    "type": "quiz", 
                    "content": content,
                    "quiz_data": quiz_result  # 구조화된 데이터도 함께 반환
                }
            else:
                # 폴백: 기본 퀴즈 생성
                fallback_q = f"다음 질문에 답하세요: {info_prompt}"
                return {"type": "quiz", "content": f"문제: {fallback_q}\n정답(모범답안): 관련 정보를 찾지 못했습니다."}
                
        except Exception as e:
            logger.error(f"퀴즈 생성 중 오류: {str(e)}")
            return {"type": "error", "message": f"퀴즈 생성 중 오류: {str(e)}"}

class BDIFlowOrchestrator:
    """BDI 플로우 오케스트레이터"""
    
    def __init__(self):
        self.input_refiner = InputRefiner()
        self.coref_resolver = CoreferenceResolver()
        self.intention_classifier = IntentionClassifier()
        self.qa_agent = QAAgent()
        self.desire_classifier = DesireClassifier()
        self.task_manager = TaskManager()
        self.belief_manager = BeliefManager()
        self.output_refiner = OutputRefiner()
        self.function_caller = FunctionCaller()
        self.static_text = StaticTextResponder()
        self.quiz_generator = QuizGenerator(self.qa_agent)
    
    def process(self, user_input: str, user_id: str = "default") -> Dict[str, Any]:
        """BDI 플로우 전체 처리"""
        logger.info(f"BDI 플로우 시작: {user_input}")
        
        try:
            # 1. Input_Refiner: 입력 메시지 정제
            refined_input = self.input_refiner.refine(user_input)
            
            # 2. Intention_Classifier: 의도 분석
            intention = self.intention_classifier.classify(refined_input)
            
            # 3. Belief_Manager: Belief 조회 및 업데이트
            self.belief_manager.update(refined_input)
            
            # 3.5 최근 대화 컨텍스트 확보 및 지시어 해소
            convo_context = None
            try:
                from .chat_ui import get_conversation_context
                convo_context = get_conversation_context(user_id)
            except Exception:
                convo_context = None
            resolved_input = self.coref_resolver.resolve(refined_input, convo_context)

            # 4. 의도에 따른 분기 처리
            if intention == "incorrect":
                result = {"type": "incorrect", "message": "입력이 불명확합니다. 구체적으로 질문이나 요청을 다시 적어주세요."}
            elif intention == "question" or intention in ["who", "when", "why", "how", "compare", "summarize"]:
                result = self.qa_agent.process(resolved_input, context=convo_context)
            elif intention == "desire":
                result = self.desire_classifier.process(resolved_input)
            elif intention == "task":
                result = self.task_manager.process(resolved_input)
            elif intention == "quiz":
                result = self.quiz_generator.process(resolved_input, context=convo_context)
            elif intention == "function":
                result = self.function_caller.call(resolved_input)
            elif intention == "static_text":
                result = self.static_text.respond(resolved_input)
            else:
                result = {"type": "error", "message": "의도를 파악할 수 없습니다."}
            
            # 5. Output_Refiner: 출력 정제
            final_answer = self.output_refiner.refine(result)
            
            # 6. 결과 반환
            return {
                "success": True,
                "intention": intention,
                "refined_input": refined_input,
                "resolved_input": resolved_input,
                "result": result,
                "final_answer": final_answer,
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"BDI 플로우 처리 실패: {str(e)}")
            return {
                "success": False,
                "error": f"시스템 오류가 발생했습니다: {str(e)}",
                "user_id": user_id
            }
