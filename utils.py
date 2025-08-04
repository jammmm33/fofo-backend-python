import os


def get_predefined_questions() -> list:
    return [
        {
            "full_text": "자신의 강점이 잘 드러난 경험 하나를 소개해주세요.",
            "short_text": "강점이 드러난 경험"
        },
        {
            "full_text": "가장 자신 있는 프로젝트 또는 작업 경험은 무엇인가요?",
            "short_text": "자신 있는 프로젝트"
        },
        {
            "full_text": "협업 중 기억에 남는 순간이나 갈등 해결 사례가 있나요?",
            "short_text": "협업 갈등 해결 사례"
        },
        {
            "full_text": "가장 힘들었지만 성장했다고 느낀 순간은 언제였나요?",
            "short_text": "가장 성장한 순간"
        }
    ]
