#!/bin/bash

# vllm과 transformers 사이의 aimv2 모델 충돌 해결 스크립트
# ovis.py 파일에서 AutoConfig.register("aimv2", AIMv2Config) 라인을 주석 처리

# 가상환경 경로 찾기
VENV_PATH=$VIRTUAL_ENV
OVIS_FILE="${VENV_PATH}/lib/python3.10/site-packages/vllm/transformers_utils/configs/ovis.py"

echo "========================================"
echo "vLLM AIMv2 충돌 해결 스크립트"
echo "========================================"

# 파일 존재 확인
if [ ! -f "$OVIS_FILE" ]; then
    echo "❌ 에러: ovis.py 파일을 찾을 수 없습니다: $OVIS_FILE"
    exit 1
fi

echo "✓ 파일 찾음: $OVIS_FILE"

# 백업 생성
BACKUP_FILE="${OVIS_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
cp "$OVIS_FILE" "$BACKUP_FILE"
echo "✓ 백업 생성: $BACKUP_FILE"

# AutoConfig.register("aimv2", AIMv2Config) 라인을 주석 처리
sed -i 's/^\(\s*\)AutoConfig\.register("aimv2", AIMv2Config)/\1# AutoConfig.register("aimv2", AIMv2Config)  # Commented out due to conflict with transformers/' "$OVIS_FILE"

# 변경 확인
if grep -q '# AutoConfig.register("aimv2", AIMv2Config)' "$OVIS_FILE"; then
    echo "✓ 성공적으로 수정되었습니다!"
    echo ""
    echo "수정된 내용:"
    grep -n "aimv2" "$OVIS_FILE" || echo "  (aimv2 관련 활성 라인 없음 - 주석 처리 완료)"
else
    echo "❌ 경고: 수정이 제대로 적용되지 않았을 수 있습니다."
    echo "파일을 확인하세요: $OVIS_FILE"
fi

echo ""
echo "========================================"
echo "완료! 이제 다시 실행해보세요."
echo "========================================"
