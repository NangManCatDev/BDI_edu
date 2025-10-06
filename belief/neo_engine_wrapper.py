# belief/neo_engine_wrapper.py
"""
순수 NEO 엔진을 사용하는 래퍼 클래스
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from ctypes import cdll, c_char_p, c_int, create_string_buffer

logger = logging.getLogger(__name__)

class NEOEngine:
    """
    순수 NEO 엔진을 사용하는 래퍼 클래스
    """
    
    def __init__(self):
        self.predicates = {}
        self.facts = []
        self.rules = []
        self.neo_executor = None
        self._initialize_neo()
    
    def _initialize_neo(self):
        """NEO 엔진 초기화 (플랫폼 무관하게 네이티브 라이브러리 사용)"""
        try:
            dll_path = os.getenv("NEO_DLL_PATH")
            if not dll_path:
                # 프로젝트 루트의 기본 파일명 시도
                root_default = os.path.join(os.path.dirname(os.path.dirname(__file__)), "libNeoDLL.so")
                if os.path.exists(root_default):
                    dll_path = root_default
            if not dll_path or not os.path.exists(dll_path):
                raise FileNotFoundError("NEO 네이티브 라이브러리(.so)를 찾을 수 없습니다. NEO_DLL_PATH를 설정하세요.")

            self.neo_executor = _CNeoExecutor(dll_path)
            logger.info(f"NEO 엔진 초기화 완료: {dll_path}")
        except Exception as e:
            logger.error(f"NEO 엔진 초기화 실패: {str(e)}")
            # 필수 조건: DLL 사용. 실패 시 예외를 그대로 올려 시스템이 인지하도록 함
            raise
    
    def pred(self, name, arity):
        """Predicate 정의"""
        # 간단한 predicate 객체 생성
        predicate = type('Predicate', (), {
            'name': name,
            'arity': arity,
            'get_arity': lambda: arity
        })()
        self.predicates[name] = predicate
        return predicate
    
    def fact(self, predicate_or_atom, args=None):
        """Fact 추가"""
        if args is None:
            # Atom을 직접 전달받은 경우
            self.facts.append(predicate_or_atom)
        else:
            # Predicate + args 전달받은 경우
            atom = type('Atom', (), {
                'predicate': predicate_or_atom,
                'arguments': args
            })()
            self.facts.append(atom)
        
        # NEO 엔진에도 추가
        if self.neo_executor:
            try:
                fact_str = self._atom_to_neo_string(predicate_or_atom if args is None else atom)
                result, output = self.neo_executor.execute_query(fact_str)
                if result != 1:
                    logger.warning(f"NEO 엔진에 fact 추가 실패: {fact_str} -> {result}, {output}")
            except Exception as e:
                logger.warning(f"NEO 엔진 fact 추가 중 오류: {str(e)}")
    
    def rule(self, head_or_clause, body_atoms=None):
        """Rule 추가"""
        if body_atoms is None:
            # Clause 객체 직접 추가
            self.rules.append(head_or_clause)
        else:
            clause = type('Clause', (), {
                '_head': head_or_clause,
                '_body': body_atoms
            })()
            self.rules.append(clause)
        
        # NEO 엔진에도 추가
        if self.neo_executor:
            try:
                rule_str = self._clause_to_neo_string(head_or_clause if body_atoms is None else clause)
                result, output = self.neo_executor.execute_query(rule_str)
                if result != 1:
                    logger.warning(f"NEO 엔진에 rule 추가 실패: {rule_str} -> {result}, {output}")
            except Exception as e:
                logger.warning(f"NEO 엔진 rule 추가 중 오류: {str(e)}")
    
    def query(self, atom):
        """Query 실행"""
        results = []
        
        # NEO 엔진을 우선 사용
        if self.neo_executor:
            try:
                query_str = self._atom_to_neo_string(atom)
                result, output = self.neo_executor.execute_query(query_str)
                if result == 1:
                    # NEO 엔진 결과를 파싱
                    results = self._parse_neo_output(output, atom)
                    if results:
                        return results
            except Exception as e:
                logger.warning(f"NEO 엔진 쿼리 실행 실패: {str(e)}")
        
        # DLL 사용이 필수이므로 실패 시 빈 결과 반환 (또는 예외)
        logger.error("NEO 엔진 쿼리 실행에 실패했습니다.")
        return []
    
    def _fallback_query(self, atom):
        """간단한 매칭 방식으로 쿼리 실행"""
        results = []
        
        # Fact 검사
        for f in self.facts:
            if hasattr(f, 'predicate') and hasattr(atom, 'predicate'):
                if f.predicate.name == atom.predicate.name:
                    subst = {}
                    matched = True
                    for fa, qa in zip(f.arguments, atom.arguments):
                        if hasattr(qa, 'name'):  # Variable인 경우
                            subst[qa.name] = fa
                        elif fa != qa:
                            matched = False
                            break
                    if matched:
                        results.append(subst)
        
        # Rule 검사
        for r in self.rules:
            if hasattr(r, '_head') and hasattr(atom, 'predicate'):
                head = r._head
                if head.predicate.name == atom.predicate.name:
                    subst = {}
                    matched = True
                    for ha, qa in zip(head.arguments, atom.arguments):
                        if hasattr(qa, 'name'):  # Variable인 경우
                            subst[qa.name] = ha
                        elif ha != qa:
                            matched = False
                            break
                    if matched:
                        results.append(subst)
        
        # 중복 제거
        unique_results = []
        for r in results:
            if r not in unique_results:
                unique_results.append(r)
        
        return unique_results
    
    def query_simple(self, query_string):
        """문자열 쿼리를 네이티브 NEO 엔진으로 실행"""
        if not self.neo_executor:
            return []
        try:
            result, output = self.neo_executor.execute_query(query_string)
            if result == 1:
                # 간단 파싱: 출력 전체를 결과 필드로 전달
                return [{
                    'predicate': query_string.split('(')[0].strip(),
                    'arguments': [],
                    'fact_string': output.strip() if output else query_string
                }]
            else:
                return []
        except Exception as e:
            logger.error(f"NEO 문자열 쿼리 실패: {str(e)}")
            return []
    
    def _atom_to_neo_string(self, atom):
        """Atom을 NEO 엔진 문자열로 변환"""
        if hasattr(atom, 'predicate') and hasattr(atom, 'arguments'):
            pred_name = atom.predicate.name
            args = []
            for arg in atom.arguments:
                if hasattr(arg, 'name'):  # Variable인 경우
                    args.append(arg.name)
                else:
                    # 상수인 경우 따옴표 제거
                    arg_str = str(arg).strip("'")
                    args.append(f"'{arg_str}'")
            return f"{pred_name}({', '.join(args)})"
        return str(atom)
    
    def _clause_to_neo_string(self, clause):
        """Clause를 NEO 엔진 문자열로 변환"""
        if hasattr(clause, '_head') and hasattr(clause, '_body'):
            head_str = self._atom_to_neo_string(clause._head)
            body_strs = [self._atom_to_neo_string(body) for body in clause._body]
            return f"{head_str} :- {', '.join(body_strs)}"
        return str(clause)
    
    def _parse_neo_output(self, output, query_atom):
        """NEO 엔진 출력을 파싱"""
        # NEO 엔진 출력 형식에 따라 파싱 로직 구현
        results = []
        
        if output and output.strip():
            # 변수 바인딩 파싱 (실제 구현은 NEO 엔진 출력 형식에 따라 달라짐)
            try:
                # 예시: "X1=value1, X2=value2" 형태의 출력을 파싱
                if "=" in output:
                    bindings = {}
                    for binding in output.split(","):
                        if "=" in binding:
                            var, value = binding.strip().split("=", 1)
                            bindings[var.strip()] = value.strip().strip("'")
                    if bindings:
                        results.append(bindings)
            except Exception as e:
                logger.warning(f"NEO 출력 파싱 실패: {str(e)}")
        
        return results
    
    def cleanup(self):
        """NEO 엔진 정리"""
        if self.neo_executor:
            try:
                self.neo_executor.cleanup()
            except Exception as e:
                logger.warning(f"NEO 엔진 정리 실패: {str(e)}")


class _CNeoExecutor:
    """ctypes 기반 NEO 네이티브 호출기"""
    def __init__(self, dll_path: str):
        self.lib = cdll.LoadLibrary(dll_path)
        # 함수 시그니처 설정 (일반적인 패턴 기반)
        # int NEO_Init();
        if hasattr(self.lib, 'NEO_Init'):
            self.lib.NEO_Init.restype = c_int
            self.lib.NEO_Init.argtypes = []
            init_rc = self.lib.NEO_Init()
            logger.info(f"NEO_Init -> {init_rc}")
        # int NEO_Exit();
        if hasattr(self.lib, 'NEO_Exit'):
            self.lib.NEO_Exit.restype = c_int
            self.lib.NEO_Exit.argtypes = []
        # int NEO_EventEngine(const char* in, char* out);
        # 일부 구현은 (in)->(ret, outbuf)에 따라 다를 수 있어 out 길이 인자 사용이 없을 수 있음
        if hasattr(self.lib, 'NEO_EventEngine'):
            self.lib.NEO_EventEngine.restype = c_int
            self.lib.NEO_EventEngine.argtypes = [c_char_p, c_char_p]

        self.max_buf = int(os.getenv('NEO_DLL_MAXBUF', '65536'))

    def execute_query(self, query: str):
        """쿼리를 네이티브 엔진에 전달하고 결과 문자열을 반환"""
        if not hasattr(self.lib, 'NEO_EventEngine'):
            raise RuntimeError('NEO_EventEngine 심볼이 없습니다.')
        in_bytes = query.encode('utf-8')
        out_buf = create_string_buffer(self.max_buf)
        rc = self.lib.NEO_EventEngine(c_char_p(in_bytes), out_buf)
        out = out_buf.value.decode('utf-8', errors='ignore') if out_buf.value else ''
        return rc, out

    def cleanup(self):
        if hasattr(self.lib, 'NEO_Exit'):
            try:
                rc = self.lib.NEO_Exit()
                logger.info(f"NEO_Exit -> {rc}")
            except Exception:
                pass

# 기존 코드와의 호환성을 위한 별칭
PrologEngine = NEOEngine