#!/usr/bin/env python3
"""
NEO 엔진 터미널 콘솔 (Linux)

기능(간단 콘솔형):
- .so(DLL) 로드 (NEO_DLL_PATH/--dll)
- .nkb 로드(:load), 쿼리 실행, 원시 출력 확인(:raw on/off)
- neoconsolekernel.py에서 쓰던 스타일 참고해 명령어 기반 REPL 지원

사용 예:
  python neo_terminal_test.py --kb sample_history.nkb --dll ./libNeoDLL.so
환경변수:
  NEO_DLL_PATH=/abs/path/libNeoDLL.so  (인자 미지정 시 사용)
  NEO_DLL_MAXBUF=65536                 (선택)
종료:
  exit, quit, :q
"""

import os
import sys
import argparse
import logging

from ctypes import cdll, c_char_p, c_int, create_string_buffer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("neo_terminal")


def ensure_dll(dll_path: str | None) -> str:
    if dll_path:
        dll_path = os.path.abspath(dll_path)
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"지정한 DLL(.so) 파일이 존재하지 않습니다: {dll_path}")
        os.environ['NEO_DLL_PATH'] = dll_path
        return dll_path

    env_path = os.getenv('NEO_DLL_PATH')
    if env_path and os.path.exists(env_path):
        return env_path

    # 프로젝트 루트의 기본 파일명 시도
    root_default = os.path.abspath(os.path.join(os.path.dirname(__file__), 'libNeoDLL.so'))
    if os.path.exists(root_default):
        os.environ['NEO_DLL_PATH'] = root_default
        return root_default

    raise FileNotFoundError("NEO_DLL_PATH 환경변수 또는 --dll 인자로 .so 경로를 지정하세요.")


def main():
    parser = argparse.ArgumentParser(description="NEO 엔진 터미널 테스트")
    parser.add_argument('--kb', type=str, default='sample_history.nkb', help='로드할 .nkb 파일 경로')
    parser.add_argument('--dll', type=str, default=None, help='NEO 네이티브 라이브러리(.so) 경로')
    args = parser.parse_args()

    try:
        dll = ensure_dll(args.dll)
        logger.info(f"NEO DLL 경로: {dll}")
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)

    kb_file = os.path.abspath(args.kb)
    if not os.path.exists(kb_file):
        logger.error(f".nkb 파일을 찾을 수 없습니다: {kb_file}")
        sys.exit(1)

    # 네이티브 로드 및 초기화
    try:
        lib = cdll.LoadLibrary(os.getenv('NEO_DLL_PATH'))
    except Exception as e:
        logger.error(f"DLL 로드 실패: {e}")
        sys.exit(1)

    # 심볼 설정
    neo_init = getattr(lib, 'NEO_Init', None)
    neo_exit = getattr(lib, 'NEO_Exit', None)
    neo_event = getattr(lib, 'NEO_EventEngine', None)
    if neo_event is None:
        logger.error('NEO_EventEngine 심볼을 찾을 수 없습니다.')
        sys.exit(1)
    if neo_init:
        neo_init.restype = c_int; neo_init.argtypes = []
        rc = neo_init(); logger.info(f"NEO_Init -> {rc}")
    neo_event.restype = c_int; neo_event.argtypes = [c_char_p, c_char_p]
    if neo_exit:
        neo_exit.restype = c_int; neo_exit.argtypes = []

    max_buf = int(os.getenv('NEO_DLL_MAXBUF', '65536'))

    def run_raw(cmd: str):
        out_buf = create_string_buffer(max_buf)
        rc = neo_event(c_char_p(cmd.encode('utf-8')), out_buf)
        out = out_buf.value.decode('utf-8', errors='ignore') if out_buf.value else ''
        return rc, out

    # 초기에 KB 로드 시도
    def load_kb(path: str) -> bool:
        if not os.path.exists(path):
            print(f"[LOAD] 파일 없음: {path}")
            return False
        # KB 로드는 파일의 각 라인을 쿼리로 전달하는 방식(엔진 규약에 따라 조정)
        ok, total = 0, 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';'):  # 주석/빈줄
                    continue
                total += 1
                rc, _ = run_raw(line)
                if rc == 1:
                    ok += 1
        print(f"[LOAD] {ok}/{total} facts loaded from {path}")
        return ok > 0

    if not load_kb(kb_file):
        logger.warning("초기 KB 로드 실패 또는 비어 있음")

    print("NEO 콘솔 준비. 쿼리 또는 명령어를 입력하세요.")
    print(":load <file>  | :dll <path>  | :raw on|off  | :help  | :q")

    show_raw = True

    # REPL 루프
    while True:
        try:
            query = input('neo> ').strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            break
        if not query:
            continue
        if query.lower() in {':q', 'exit', 'quit'}:
            break

        # 명령어 처리
        if query.startswith(':'):
            parts = query.split()
            cmd = parts[0]
            if cmd == ':load' and len(parts) >= 2:
                load_kb(parts[1])
                continue
            if cmd == ':dll' and len(parts) >= 2:
                try:
                    os.environ['NEO_DLL_PATH'] = os.path.abspath(parts[1])
                    print(f"[DLL] 재설정됨: {os.environ['NEO_DLL_PATH']}")
                except Exception as e:
                    print(f"[DLL] 실패: {e}")
                continue
            if cmd == ':raw' and len(parts) >= 2:
                show_raw = (parts[1].lower() == 'on')
                print(f"[RAW] {'ON' if show_raw else 'OFF'}")
                continue
            if cmd == ':help':
                print(":load <file>  | :dll <path>  | :raw on|off  | :q")
                continue
            print("알 수 없는 명령어입니다. :help 참고")
            continue

        # 일반 쿼리: NEO_EventEngine 원시 실행
        try:
            rc, out = run_raw(query)
            print(f"[RC] {rc}")
            # 엔진이 별도 파일(output.txt)에 결과를 쓰는 구현을 지원
            file_preview = None
            try:
                if os.path.exists('output.txt'):
                    with open('output.txt', 'r', encoding='utf-8', errors='ignore') as f:
                        file_preview = f.read()
                        if len(file_preview) > 2000:
                            file_preview = file_preview[:2000] + '...'
            except Exception:
                file_preview = None

            preview = out if out else "(no output)"
            if len(preview) > 2000:
                preview = preview[:2000] + '...'
            if show_raw:
                if file_preview:
                    print("[FILE] output.txt:")
                    print(file_preview)
                print("[RAW]")
                print(preview)
        except Exception as e:
            logger.error(f"쿼리 실행 실패: {e}")

    # 종료 정리
    try:
        if neo_exit:
            rc = neo_exit(); logger.info(f"NEO_Exit -> {rc}")
    except Exception:
        pass

    logger.info("종료합니다.")


if __name__ == '__main__':
    main()


