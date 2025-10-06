# belief/integrated_kb_builder.py
"""
통합된 지식베이스 빌더 - 텍스트 파일을 NEO 지식베이스로 변환하고 로드
"""

import os
import logging
from typing import List, Dict, Any, Optional
from .text_to_kb_converter import TextToKBConverter
from .neo_kb_loader import NEOKBLoader
from .neo_engine_wrapper import NEOEngine

logger = logging.getLogger(__name__)

class IntegratedKBBuilder:
    """
    텍스트 파일을 NEO 지식베이스로 변환하고 로드하는 통합 빌더
    """
    
    def __init__(self):
        self.converter = TextToKBConverter()
        self.loader = NEOKBLoader()
        self.engine = None
    
    def build_from_text_file(self, text_file_path: str, kb_output_path: str = None) -> bool:
        """
        텍스트 파일을 NEO 지식베이스로 변환하고 로드
        
        Args:
            text_file_path: 입력 텍스트 파일 경로
            kb_output_path: 출력 KB 파일 경로 (기본값: 자동 생성)
            
        Returns:
            빌드 성공 여부
        """
        try:
            logger.info(f"통합 KB 빌드 시작: {text_file_path}")
            
            # 1단계: 텍스트를 KB로 변환
            if kb_output_path is None:
                base_name = os.path.splitext(text_file_path)[0]
                kb_output_path = f"{base_name}.kb"
            
            logger.info("1단계: 텍스트를 KB로 변환 중...")
            conversion_success = self.converter.convert_text_file(text_file_path, kb_output_path)
            
            if not conversion_success:
                logger.error("텍스트 변환 실패")
                return False
            
            logger.info("✅ 텍스트 변환 완료")
            
            # 2단계: KB 파일 검증
            logger.info("2단계: KB 파일 검증 중...")
            validation_success = self.converter.validate_kb_file(kb_output_path)
            
            if not validation_success:
                logger.error("KB 파일 검증 실패")
                return False
            
            logger.info("✅ KB 파일 검증 완료")
            
            # 3단계: KB 파일을 NEO 엔진에 로드
            logger.info("3단계: KB 파일을 NEO 엔진에 로드 중...")
            load_success = self.loader.load_kb_file(kb_output_path)
            
            if not load_success:
                logger.error("KB 파일 로드 실패")
                return False
            
            logger.info("✅ KB 파일 로드 완료")
            
            # 4단계: 엔진 설정
            self.engine = self.loader.get_engine()
            
            # 통계 출력
            stats = self.loader.get_stats()
            logger.info(f"최종 통계: {stats}")
            
            logger.info("🎉 통합 KB 빌드 완료!")
            return True
            
        except Exception as e:
            logger.error(f"통합 KB 빌드 실패: {str(e)}")
            return False
    
    def build_from_text_directory(self, text_dir: str, kb_output_dir: str = None) -> Dict[str, bool]:
        """
        텍스트 디렉토리를 NEO 지식베이스로 변환하고 로드
        
        Args:
            text_dir: 입력 텍스트 디렉토리 경로
            kb_output_dir: 출력 KB 디렉토리 경로 (기본값: 자동 생성)
            
        Returns:
            파일별 빌드 성공 여부 딕셔너리
        """
        try:
            logger.info(f"통합 KB 디렉토리 빌드 시작: {text_dir}")
            
            if kb_output_dir is None:
                kb_output_dir = text_dir + "_kb"
            
            # 1단계: 텍스트 디렉토리를 KB 디렉토리로 변환
            logger.info("1단계: 텍스트 디렉토리를 KB 디렉토리로 변환 중...")
            conversion_results = self.converter.convert_directory(text_dir, kb_output_dir)
            
            successful_conversions = [f for f, success in conversion_results.items() if success]
            logger.info(f"✅ 변환 완료: {len(successful_conversions)}개 파일")
            
            # 2단계: 변환된 KB 파일들을 NEO 엔진에 로드
            logger.info("2단계: KB 파일들을 NEO 엔진에 로드 중...")
            load_results = self.loader.load_directory(kb_output_dir)
            
            # 3단계: 엔진 설정
            self.engine = self.loader.get_engine()
            
            # 최종 결과 계산
            final_results = {}
            for file in conversion_results:
                final_results[file] = conversion_results[file] and load_results.get(file, False)
            
            # 통계 출력
            stats = self.loader.get_stats()
            logger.info(f"최종 통계: {stats}")
            
            successful_builds = sum(1 for success in final_results.values() if success)
            logger.info(f"🎉 통합 KB 디렉토리 빌드 완료: {successful_builds}개 파일 성공")
            
            return final_results
            
        except Exception as e:
            logger.error(f"통합 KB 디렉토리 빌드 실패: {str(e)}")
            return {}
    
    def get_engine(self) -> Optional[NEOEngine]:
        """
        빌드된 NEO 엔진 반환
        
        Returns:
            빌드된 NEO 엔진 또는 None
        """
        return self.engine
    
    def get_stats(self) -> Dict[str, Any]:
        """
        빌드된 지식베이스 통계 정보 반환
        
        Returns:
            통계 정보 딕셔너리
        """
        if self.engine is None:
            return {"status": "not_built"}
        
        return self.loader.get_stats()
    
    def query(self, query_text: str) -> List[Dict[str, Any]]:
        """
        빌드된 지식베이스에 쿼리 실행
        
        Args:
            query_text: 쿼리 텍스트
            
        Returns:
            쿼리 결과 리스트
        """
        if self.engine is None:
            logger.error("지식베이스가 빌드되지 않음")
            return []
        
        try:
            # 간단한 쿼리 객체 생성 (실제로는 nl2kqml을 사용해야 함)
            # 여기서는 예시로 person 쿼리 생성
            if "person" in query_text.lower():
                person_pred = self.engine.predicates.get('person')
                if person_pred:
                    # person(X1, X2, X3, X4) 쿼리 생성
                    query_atom = type('Atom', (), {
                        'predicate': person_pred,
                        'arguments': [
                            type('Variable', (), {'name': 'X1'})(),
                            type('Variable', (), {'name': 'X2'})(),
                            type('Variable', (), {'name': 'X3'})(),
                            type('Variable', (), {'name': 'X4'})()
                        ]
                    })()
                    
                    return self.engine.query(query_atom)
            
            return []
            
        except Exception as e:
            logger.error(f"쿼리 실행 실패: {str(e)}")
            return []

# 사용 예시
if __name__ == "__main__":
    builder = IntegratedKBBuilder()
    
    # 단일 텍스트 파일 빌드
    success = builder.build_from_text_file("sample_history.txt")
    if success:
        print("✅ 단일 파일 빌드 성공")
        stats = builder.get_stats()
        print(f"통계: {stats}")
        
        # 쿼리 테스트
        results = builder.query("person")
        print(f"쿼리 결과: {results}")
    else:
        print("❌ 단일 파일 빌드 실패")
    
    # 텍스트 디렉토리 빌드
    results = builder.build_from_text_directory("text_files/")
    print(f"디렉토리 빌드 결과: {results}")
