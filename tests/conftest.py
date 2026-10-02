import os
import sys
from pathlib import Path

os.environ["HEARME_NO_SELECTED"] = "1"   # 단위 테스트는 배포 구성(selected_config.yaml)과 무관하게 기본 정책을 검사
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
