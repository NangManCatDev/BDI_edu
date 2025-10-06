# KB Generator - 독립적인 텍스트 → NEO KB 변환 모듈

## 개요
자연어 텍스트 파일을 NEO 지식베이스 형식으로 변환하는 독립적인 모듈입니다.

## 설치
```bash
cd kb_generator
pip install -r requirements.txt
```

## 환경 설정
```bash
export OPENAI_API_KEY='your-api-key'
```

## 사용법
```bash
python txt_to_kb.py <입력파일.txt> <출력파일.nkb>
```

## 예시
```bash
# 샘플 텍스트를 KB로 변환
python txt_to_kb.py sample.txt history.nkb
```

## 출력 형식
- `event(Name, Year, Desc)` - 사건 정보
- `person(Name, Birth, Death, Role)` - 인물 정보
- `cause(Event, Result)` - 인과관계

## 특징
- **독립적**: 전체 BDI 프레임워크와 분리
- **사전 생성**: 프레임워크 실행 전 KB 파일 생성
- **최적화**: 실행 시마다 변환하지 않음
- **청크 처리**: 긴 텍스트를 자동으로 분할 처리
