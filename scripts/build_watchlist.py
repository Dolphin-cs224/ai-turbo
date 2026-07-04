import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


# -----------------------------
# 설정
# -----------------------------
KOSPI_TOP_PERCENT = 0.20
KOSDAQ_TOP_PERCENT = 0.01

OUTPUT_PATH = Path("data/watchlist.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# -----------------------------
# 제외 종목 필터
# ETF, ETN, 스팩, 우선주, 리츠 등 자동 제외
# -----------------------------
EXCLUDE_KEYWORDS = [
    # ETF / ETN 운용사 브랜드
    "KODEX", "TIGER", "ACE", "RISE", "KBSTAR", "SOL",
    "HANARO", "ARIRANG", "KOSEF", "TIMEFOLIO", "KoAct",
    "히어로즈", "마이티", "TREX", "FOCUS", "PLUS",

    # ETF/ETN 상품명에 자주 들어가는 단어
    "ETF", "ETN", "액티브", "레버리지", "인버스",
    "선물", "채권", "국채", "회사채", "TDF", "TR",

    # 스팩 / 리츠
    "스팩", "SPAC", "기업인수목적", "리츠", "REIT",
]


def is_excluded_stock(name: str) -> bool:
    """
    투자 유니버스에서 제외할 종목인지 판단한다.
    ETF, ETN, 스팩, 우선주, 리츠 등을 제외한다.
    """
    if not isinstance(name, str):
        return True

    clean_name = name.strip()

    # 키워드 기반 제외
    for keyword in EXCLUDE_KEYWORDS:
        if keyword in clean_name:
            return True

    # 우선주 제외
    # 예: 삼성전자우, 현대차우, LG화학우, 하이트진로2우B
    if re.search(r"(우|우B|우선주)$", clean_name):
        return True

    # 스팩 이름 패턴 추가 제외
    # 예: 하나스팩34호, 미래에셋비전스팩7호
    if re.search(r"스팩\d*호", clean_name):
        return True

    return False

def get_naver_market_sum(sosok: int, market_name: str, max_pages: int = 60):
    """
    네이버 금융 시가총액 순위에서 종목명, 종목코드, 시가총액을 가져온다.

    sosok=0 : KOSPI
    sosok=1 : KOSDAQ
    """
    result = []

    for page in range(1, max_pages + 1):
        url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}"

        response = requests.get(url, headers=HEADERS)
        response.encoding = "euc-kr"

        # 1. BeautifulSoup으로 종목명 + 종목코드 추출
        soup = BeautifulSoup(response.text, "html.parser")

        code_rows = []

        for link in soup.select("a.tltle"):
            name = link.get_text(strip=True)
            href = link.get("href", "")

            match = re.search(r"code=(\d+)", href)

            if match:
                code_rows.append({
                    "name": name,
                    "code": match.group(1)
                })

        if not code_rows:
            break

        code_df = pd.DataFrame(code_rows)

        # 2. pandas.read_html로 시가총액 표 추출
        tables = pd.read_html(response.text)

        market_tables = [
            table for table in tables
            if "종목명" in table.columns and "시가총액" in table.columns
        ]

        if not market_tables:
            break

        table_df = market_tables[0]
        table_df = table_df[["종목명", "시가총액"]].dropna()
        table_df = table_df.rename(columns={
            "종목명": "name",
            "시가총액": "market_cap"
        })

        # 3. 종목명 기준으로 코드와 시가총액 합치기
        page_df = pd.merge(code_df, table_df, on="name", how="inner")
        page_df["market"] = market_name

        if page_df.empty:
            break

        result.append(page_df)

    if not result:
        raise ValueError(f"{market_name} 데이터를 가져오지 못했습니다.")

    df = pd.concat(result, ignore_index=True)
    df = df.drop_duplicates(subset=["code"]).reset_index(drop=True)

    return df

def remove_preferred_stocks(df: pd.DataFrame):
    """
    우선주 제외
    예: 삼성전자우, 현대차2우B 등
    """
    def is_preferred(name):
        return (
            name.endswith("우")
            or "우B" in name
            or "2우" in name
            or "3우" in name
        )

    return df[~df["name"].apply(is_preferred)].copy()


def classify_theme(name: str):
    """
    종목명 기반 테마 자동 분류 함수
    추후 OpenAI API 또는 뉴스 분석 기반 분류로 고도화 가능
    """
    if not isinstance(name, str):
        return "미분류"

    stock_name = name.strip()
    upper_name = stock_name.upper()

    # 시가총액 상위 종목 중 키워드만으로 분류가 어려운 종목 수동 보정
    manual_theme_map = {
        "삼성전기": "전자부품/PCB",
        "LIG디펜스앤에어로스페이스": "방산",
        "삼성에스디에스": "IT서비스",
        "에이피알": "화장품/소비재",
        "현대오토에버": "IT서비스",
        "삼성증권": "금융",
        "삼성에피스홀딩스": "바이오",
        "한국타이어앤테크놀로지": "자동차",
        "이수페타시스": "전자부품/PCB",
        "한화": "지주/복합",
        "코웨이": "소비재",
        "삼성카드": "금융",
        "HD건설기계": "기계/산업재",
        "NC": "게임",
        "맥쿼리인프라": "인프라/배당",
        "한화엔진": "조선",
        "KCC": "건설/건자재",
        "OCI홀딩스": "정유/화학",
        "한화생명": "금융",
        "영원무역": "패션/의류",
        "포스코DX": "IT서비스",
        "강원랜드": "레저/카지노",
        "한올바이오파마": "바이오",
        "한국가스공사": "원전/에너지",
        "F&F": "패션/의류",
        "서울보증보험": "금융",
        "한솔케미칼": "AI반도체",
        "효성": "지주/복합",
        "iM금융지주": "금융",
        "현대엘리베이터": "기계/산업재",
        "신영증권": "금융",
        "미래에셋생명": "금융",
        "에스원": "보안",
        "달바글로벌": "화장품/소비재",
        "동서": "식품",
        "한화비전": "보안",
        "미스토홀딩스": "지주/복합",
        "한국앤컴퍼니": "자동차",
        "코리안리": "금융",
        "영원무역홀딩스": "패션/의류",
        "현대지에프홀딩스": "지주/복합",
        "케이뱅크": "금융",
        "DN오토모티브": "자동차",
        "한미사이언스": "바이오",
        "제일기획": "광고/미디어",
        "케이씨텍": "AI반도체",
        "이수스페셜티케미컬": "2차전지",
        "시프트업": "게임",
        "대한조선": "조선",
        "코리아써키트": "전자부품/PCB",
        "코오롱티슈진": "바이오",
        "펩트론": "바이오",
    }

    if stock_name in manual_theme_map:
        return manual_theme_map[stock_name]
    
    theme_rules = [
        (
            "AI반도체",
            [
                "삼성전자", "SK하이닉스", "한미반도체", "리노공업", "HPSP",
                "이오테크닉스", "주성엔지니어링", "원익IPS", "솔브레인",
                "DB하이텍", "ISC", "하나마이크론", "티씨케이", "피에스케이",
                "유진테크", "고영", "파크시스템스", "심텍", "대덕전자"
            ]
        ),
        (
            "전력인프라",
            [
                "HD현대일렉트릭", "LS ELECTRIC", "효성중공업", "대한전선",
                "가온전선", "일진전기", "제룡전기", "산일전기", "LS",
                "세명전기"
            ]
        ),
        (
            "조선",
            [
                "HD현대중공업", "삼성중공업", "한화오션", "HD한국조선해양",
                "현대미포조선", "HD현대마린엔진", "HJ중공업"
            ]
        ),
        (
            "방산",
            [
                "한화에어로스페이스", "LIG넥스원", "현대로템", "한국항공우주",
                "한화시스템", "풍산", "SNT다이내믹스", "휴니드"
            ]
        ),
        (
            "바이오",
            [
                "삼성바이오로직스", "셀트리온", "유한양행", "한미약품",
                "HLB", "알테오젠", "리가켐바이오", "SK바이오팜",
                "SK바이오사이언스", "에이비엘바이오", "보로노이",
                "종근당", "녹십자", "대웅제약", "삼천당제약",
                "휴젤", "메디톡스", "파마리서치"
            ]
        ),
        (
            "자동차",
            [
                "현대차", "기아", "현대모비스", "HL만도", "현대위아",
                "한온시스템", "에스엘", "성우하이텍", "화신", "서연이화"
            ]
        ),
        (
            "2차전지",
            [
                "LG에너지솔루션", "삼성SDI", "포스코퓨처엠", "에코프로",
                "에코프로비엠", "엘앤에프", "금양", "SK아이이테크놀로지",
                "천보", "더블유씨피", "나노신소재", "엔켐",
                "코스모신소재", "롯데에너지머티리얼즈"
            ]
        ),
        (
            "금융",
            [
                "KB금융", "신한지주", "하나금융지주", "우리금융지주",
                "메리츠금융지주", "삼성생명", "삼성화재", "DB손해보험",
                "현대해상", "미래에셋증권", "한국금융지주", "키움증권",
                "NH투자증권", "기업은행", "BNK금융지주", "JB금융지주"
            ]
        ),
        (
            "플랫폼/인터넷",
            [
                "NAVER", "카카오", "카카오뱅크", "카카오페이",
                "더존비즈온", "SOOP"
            ]
        ),
        (
            "엔터/미디어",
            [
                "하이브", "JYP", "에스엠", "와이지엔터테인먼트",
                "CJ ENM", "스튜디오드래곤", "콘텐트리중앙"
            ]
        ),
        (
            "화장품/소비재",
            [
                "아모레퍼시픽", "LG생활건강", "코스맥스", "한국콜마",
                "실리콘투", "브이티", "클리오", "토니모리", "애경산업"
            ]
        ),
        (
            "원전/에너지",
            [
                "두산에너빌리티", "한전기술", "한전KPS", "한국전력",
                "지역난방공사", "비에이치아이", "우진"
            ]
        ),
        (
            "로봇/AI",
            [
                "두산로보틱스", "레인보우로보틱스", "로보티즈",
                "유진로봇", "뉴로메카", "셀바스AI", "폴라리스AI",
                "마음AI", "솔트룩스"
            ]
        ),
        (
            "정유/화학",
            [
                "SK이노베이션", "S-Oil", "GS", "롯데케미칼",
                "한화솔루션", "금호석유", "LG화학", "대한유화"
            ]
        ),
        (
            "철강/소재",
            [
                "POSCO홀딩스", "현대제철", "고려아연", "영풍",
                "세아베스틸지주", "포스코인터내셔널"
            ]
        ),
        (
            "건설/건자재",
            [
                "현대건설", "삼성E&A", "DL이앤씨", "GS건설",
                "HDC현대산업개발", "대우건설", "한일시멘트"
            ]
        ),
        (
            "게임",
            [
                "크래프톤", "넷마블", "엔씨소프트", "펄어비스",
                "카카오게임즈", "위메이드", "컴투스"
            ]
        ),
        (
            "운송/항공",
            [
                "대한항공", "아시아나항공", "한진칼", "HMM",
                "팬오션", "CJ대한통운", "현대글로비스", "제주항공"
            ]
        ),
        (
            "식품",
            [
                "CJ제일제당", "오리온", "농심", "삼양식품",
                "하이트진로", "롯데칠성", "대상", "동원F&B"
            ]
        ),
        (
            "유통",
            [
                "이마트", "롯데쇼핑", "현대백화점", "신세계",
                "GS리테일", "BGF리테일", "호텔신라"
            ]
        ),
        (
            "통신",
            [
                "SK텔레콤", "KT", "LG유플러스"
            ]
        ),
        (
            "지주/복합",
            [
                "삼성물산", "SK스퀘어", "SK", "LG", "두산",
                "CJ", "HD현대", "롯데지주"
            ]
        ),
    ]

    for theme, keywords in theme_rules:
        for keyword in keywords:
            if keyword.upper() in upper_name:
                return theme

    return "미분류"


def build_watchlist():
    print("네이버 금융에서 KOSPI 시가총액 데이터를 가져오는 중...")
    kospi = get_naver_market_sum(sosok=0, market_name="KOSPI")

    print("네이버 금융에서 KOSDAQ 시가총액 데이터를 가져오는 중...")
    kosdaq = get_naver_market_sum(sosok=1, market_name="KOSDAQ")


    # ETF, ETN, 스팩, 우선주, 리츠 등 제외
    kospi = kospi[~kospi["name"].apply(is_excluded_stock)].copy()
    kosdaq = kosdaq[~kosdaq["name"].apply(is_excluded_stock)].copy()
    
    # 우선주 제외
    kospi = remove_preferred_stocks(kospi)
    kosdaq = remove_preferred_stocks(kosdaq)

    # 상위 비율 계산
    kospi_top_count = max(1, int(len(kospi) * KOSPI_TOP_PERCENT))
    kosdaq_top_count = max(1, int(len(kosdaq) * KOSDAQ_TOP_PERCENT))

    kospi_top = kospi.head(kospi_top_count).copy()
    kosdaq_top = kosdaq.head(kosdaq_top_count).copy()

    kospi_top["rank"] = range(1, len(kospi_top) + 1)
    kosdaq_top["rank"] = range(1, len(kosdaq_top) + 1)

    kospi_top["universe_rule"] = "KOSPI_TOP_20_PERCENT"
    kosdaq_top["universe_rule"] = "KOSDAQ_TOP_1_PERCENT"

    universe = pd.concat([kospi_top, kosdaq_top], ignore_index=True)

    universe["code"] = universe["code"].astype(str).str.zfill(6)
    universe["theme"] = universe["name"].apply(classify_theme)
    universe["sub_theme"] = "미분류"
    universe["max_weight"] = 0.10
    universe["active"] = "Y"
    universe["memo"] = ""

    watchlist = universe[
        [
            "code",
            "name",
            "theme",
            "sub_theme",
            "market",
            "rank",
            "market_cap",
            "max_weight",
            "active",
            "universe_rule",
            "memo"
        ]
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    watchlist.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("watchlist.csv 자동 생성 완료")
    print(f"KOSPI 전체 종목 수: {len(kospi)}")
    print(f"KOSPI 상위 20% 종목 수: {len(kospi_top)}")
    print(f"KOSDAQ 전체 종목 수: {len(kosdaq)}")
    print(f"KOSDAQ 상위 1% 종목 수: {len(kosdaq_top)}")
    print(f"최종 관심종목 수: {len(watchlist)}")
    print(f"저장 위치: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_watchlist()