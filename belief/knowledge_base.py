from .neo_engine_wrapper import NEOEngine as PrologEngine
from .integrated_kb_builder import IntegratedKBBuilder
import os
import logging

logger = logging.getLogger(__name__)

def build_kb():
    """
    지식베이스 빌드 - 텍스트 파일이 있으면 변환, 없으면 기본 데이터 사용
    """
    # 텍스트 파일 경로들 확인
    text_files = [
        "data/history.txt",
        "data/korean_history.txt", 
        "text_files/history.txt",
        "sample_history.txt"
    ]
    
    # 텍스트 파일이 있으면 변환 시도
    for text_file in text_files:
        if os.path.exists(text_file):
            logger.info(f"텍스트 파일 발견: {text_file}")
            builder = IntegratedKBBuilder()
            success = builder.build_from_text_file(text_file)
            if success:
                logger.info("✅ 텍스트 파일에서 지식베이스 빌드 성공")
                return builder.get_engine()
            else:
                logger.warning("텍스트 파일 변환 실패, 기본 데이터 사용")
            break
    
    # 텍스트 파일이 없거나 변환 실패 시 기본 데이터 사용
    logger.info("기본 지식베이스 데이터 사용")
    eng = PrologEngine()

    # === Predicates ===
    event = eng.pred("event", 3)    # event(Name, Year, Desc)
    person = eng.pred("person", 4)  # person(Name, Birth, Death, Role)
    cause = eng.pred("cause", 2)    # cause(Event, Result)

    # === Events ===
    eng.fact(event, ["'삼국통일'", "'668'", "'신라가 삼국을 통일함'"])
    eng.fact(event, ["'훈민정음반포'", "'1446'", "'세종이 훈민정음을 반포함'"])
    eng.fact(event, ["'임진왜란'", "'1592'", "'왜군이 조선을 침략함'"])
    eng.fact(event, ["'한글창제'", "'1443'", "'세종이 한글을 창제함'"])
    eng.fact(event, ["'동학농민운동'", "'1894'", "'동학을 바탕으로 한 농민 봉기'"])
    eng.fact(event, ["'3.1운동'", "'1919'", "'일제강점기 독립운동'"])
    eng.fact(event, ["'광복절'", "'1945'", "'일본으로부터 해방'"])

    # === Persons ===
    eng.fact(person, ["'세종'", "'1397'", "'1450'", "'조선의 4대 왕'"])
    eng.fact(person, ["'세종대왕'", "'1397'", "'1450'", "'조선의 4대 왕'"])
    eng.fact(person, ["'이순신'", "'1545'", "'1598'", "'조선 수군 장군'"])
    eng.fact(person, ["'김구'", "'1876'", "'1949'", "'독립운동가'"])
    eng.fact(person, ["'안중근'", "'1879'", "'1910'", "'독립운동가'"])
    eng.fact(person, ["'윤봉길'", "'1908'", "'1932'", "'독립운동가'"])
    eng.fact(person, ["'이성계'", "'1335'", "'1408'", "'조선의 건국자'"])
    eng.fact(person, ["'정도전'", "'1342'", "'1398'", "'조선의 개국공신'"])
    eng.fact(person, ["'신사임당'", "'1504'", "'1551'", "'조선의 여성 예술가'"])
    eng.fact(person, ["'이황'", "'1501'", "'1570'", "'조선의 성리학자'"])
    eng.fact(person, ["'이이'", "'1536'", "'1584'", "'조선의 성리학자'"])

    # === Causes ===
    eng.fact(cause, ["'동학농민운동'", "'갑오개혁'"])

    return eng
