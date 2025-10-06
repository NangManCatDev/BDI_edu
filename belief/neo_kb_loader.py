# belief/neo_kb_loader.py
"""
NEO 지식베이스 파일을 로드하고 관리하는 모듈
"""

import os
import logging
from typing import List, Dict, Any, Optional
from .neo_engine_wrapper import NEOEngine

logger = logging.getLogger(__name__)

class NEOKBLoader:
    """
    NEO 지식베이스 파일을 로드하고 관리하는 클래스
    """
    
    def __init__(self):
        self.engine = NEOEngine()
        self.loaded_files = []
    
    def load_kb_file(self, kb_file_path: str) -> bool:
        """
        KB 파일을 NEO 엔진에 로드
        
        Args:
            kb_file_path: 로드할 KB 파일 경로
            
        Returns:
            로드 성공 여부
        """
        try:
            logger.info(f"KB 파일 로드 시작: {kb_file_path}")
            
            if not os.path.exists(kb_file_path):
                logger.error(f"KB 파일이 존재하지 않음: {kb_file_path}")
                return False
            
            with open(kb_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            loaded_facts = 0
            loaded_rules = 0
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # 빈 줄이나 주석 건너뛰기
                if not line or line.startswith(';'):
                    continue
                
                try:
                    # Fact 또는 Rule 파싱
                    if ':-' in line:
                        # Rule 파싱
                        success = self._parse_rule(line)
                        if success:
                            loaded_rules += 1
                    else:
                        # Fact 파싱
                        success = self._parse_fact(line)
                        if success:
                            loaded_facts += 1
                        else:
                            logger.warning(f"라인 {line_num}: Fact 파싱 실패 - {line}")
                
                except Exception as e:
                    logger.warning(f"라인 {line_num} 처리 실패: {str(e)} - {line}")
                    continue
            
            logger.info(f"KB 파일 로드 완료: {loaded_facts}개 사실, {loaded_rules}개 규칙")
            self.loaded_files.append(kb_file_path)
            return True
            
        except Exception as e:
            logger.error(f"KB 파일 로드 실패: {str(e)}")
            return False
    
    def _parse_fact(self, line: str) -> bool:
        """
        Fact 라인을 파싱하여 엔진에 추가
        
        Args:
            line: 파싱할 Fact 라인
            
        Returns:
            파싱 성공 여부
        """
        try:
            # person('세종', '1397', '1450', '조선의 4대 왕') 형식 파싱
            if '(' not in line or ')' not in line:
                return False
            
            predicate_name = line.split('(')[0].strip()
            args_part = line.split('(')[1].split(')')[0]
            
            # 인수 파싱
            args = self._parse_arguments(args_part)
            
            if not args:
                return False
            
            # Predicate 가져오기 또는 생성
            if predicate_name not in self.engine.predicates:
                predicate = self.engine.pred(predicate_name, len(args))
            else:
                predicate = self.engine.predicates[predicate_name]
            
            # Fact 추가
            self.engine.fact(predicate, args)
            return True
            
        except Exception as e:
            logger.warning(f"Fact 파싱 실패: {str(e)} - {line}")
            return False
    
    def _parse_rule(self, line: str) -> bool:
        """
        Rule 라인을 파싱하여 엔진에 추가
        
        Args:
            line: 파싱할 Rule 라인
            
        Returns:
            파싱 성공 여부
        """
        try:
            # head :- body1, body2 형식 파싱
            if ':-' not in line:
                return False
            
            head_part, body_part = line.split(':-', 1)
            head_part = head_part.strip()
            body_part = body_part.strip()
            
            # Head 파싱
            head_atom = self._parse_atom(head_part)
            if not head_atom:
                return False
            
            # Body 파싱
            body_atoms = []
            for body_item in body_part.split(','):
                body_item = body_item.strip()
                body_atom = self._parse_atom(body_item)
                if body_atom:
                    body_atoms.append(body_atom)
            
            if not body_atoms:
                return False
            
            # Rule 추가
            self.engine.rule(head_atom, body_atoms)
            return True
            
        except Exception as e:
            logger.warning(f"Rule 파싱 실패: {str(e)} - {line}")
            return False
    
    def _parse_atom(self, atom_str: str) -> Optional[Any]:
        """
        Atom 문자열을 파싱하여 Atom 객체 생성
        
        Args:
            atom_str: 파싱할 Atom 문자열
            
        Returns:
            파싱된 Atom 객체 또는 None
        """
        try:
            if '(' not in atom_str or ')' not in atom_str:
                return None
            
            predicate_name = atom_str.split('(')[0].strip()
            args_part = atom_str.split('(')[1].split(')')[0]
            
            args = self._parse_arguments(args_part)
            if not args:
                return None
            
            # Predicate 가져오기 또는 생성
            if predicate_name not in self.engine.predicates:
                predicate = self.engine.pred(predicate_name, len(args))
            else:
                predicate = self.engine.predicates[predicate_name]
            
            # Atom 객체 생성
            return type('Atom', (), {
                'predicate': predicate,
                'arguments': args
            })()
            
        except Exception as e:
            logger.warning(f"Atom 파싱 실패: {str(e)} - {atom_str}")
            return None
    
    def _parse_arguments(self, args_str: str) -> List[str]:
        """
        인수 문자열을 파싱하여 인수 리스트 생성
        
        Args:
            args_str: 파싱할 인수 문자열
            
        Returns:
            파싱된 인수 리스트
        """
        args = []
        current_arg = ""
        in_quotes = False
        quote_char = None
        
        i = 0
        while i < len(args_str):
            char = args_str[i]
            
            if char in ['"', "'"] and not in_quotes:
                in_quotes = True
                quote_char = char
                current_arg += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current_arg += char
            elif char == ',' and not in_quotes:
                args.append(current_arg.strip())
                current_arg = ""
            else:
                current_arg += char
            
            i += 1
        
        if current_arg.strip():
            args.append(current_arg.strip())
        
        return args
    
    def load_directory(self, kb_dir: str) -> Dict[str, bool]:
        """
        디렉토리 내 모든 KB 파일을 로드
        
        Args:
            kb_dir: KB 파일들이 있는 디렉토리 경로
            
        Returns:
            파일별 로드 성공 여부 딕셔너리
        """
        results = {}
        
        if not os.path.exists(kb_dir):
            logger.error(f"KB 디렉토리가 존재하지 않음: {kb_dir}")
            return results
        
        # KB 파일 찾기
        kb_files = []
        for file in os.listdir(kb_dir):
            if file.endswith('.kb'):
                kb_files.append(file)
        
        logger.info(f"로드할 KB 파일 {len(kb_files)}개 발견")
        
        # 각 파일 로드
        for file in kb_files:
            file_path = os.path.join(kb_dir, file)
            logger.info(f"로드 중: {file}")
            
            success = self.load_kb_file(file_path)
            results[file] = success
            
            if success:
                logger.info(f"✅ {file} 로드 성공")
            else:
                logger.error(f"❌ {file} 로드 실패")
        
        return results
    
    def get_engine(self) -> NEOEngine:
        """
        로드된 NEO 엔진 반환
        
        Returns:
            로드된 NEO 엔진
        """
        return self.engine
    
    def get_stats(self) -> Dict[str, Any]:
        """
        로드된 지식베이스 통계 정보 반환
        
        Returns:
            통계 정보 딕셔너리
        """
        return {
            "loaded_files": len(self.loaded_files),
            "total_facts": len(self.engine.facts),
            "total_rules": len(self.engine.rules),
            "predicates": list(self.engine.predicates.keys()),
            "file_list": self.loaded_files
        }

# 사용 예시
if __name__ == "__main__":
    loader = NEOKBLoader()
    
    # 단일 KB 파일 로드
    success = loader.load_kb_file("sample.kb")
    if success:
        print("✅ KB 파일 로드 성공")
        stats = loader.get_stats()
        print(f"통계: {stats}")
    else:
        print("❌ KB 파일 로드 실패")
    
    # 디렉토리 로드
    results = loader.load_directory("kb_files/")
    print(f"로드 결과: {results}")
