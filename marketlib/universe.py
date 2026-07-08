"""
0050 (元大台灣50 ETF) constituent stock codes — manually-maintained snapshot.

IMPORTANT: This list is a best-effort approximation of the Taiwan 50 Index
constituents. It MUST be periodically verified against the official 0050
holdings published by Yuanta Securities Investment Trust Co., Ltd.
(https://www.yuantaetfs.com/product/detail/0050/proportion).
Failure to keep this list current may cause the screener to miss newly
added large-caps or include stocks that have been removed from the index.
"""

# As-of date for this snapshot — update whenever the list is refreshed.
TW50_ASOF: str = "2026-07-08"

# Codes only; human-readable names are fetched from market data at runtime.
# NOTE: This list needs periodic verification against the live 0050 holdings.
TW50_CONSTITUENTS: list[str] = [
    "2330",  # TSMC
    "2317",  # Hon Hai
    "2454",  # MediaTek
    "2308",  # Delta Electronics
    "2382",  # Quanta Computer
    "2412",  # Chunghwa Telecom
    "2881",  # Fubon Financial
    "2882",  # Cathay Financial
    "2891",  # CTBC Financial
    "2303",  # United Microelectronics
    "2886",  # Mega Financial
    "2884",  # E.Sun Financial
    "1301",  # Formosa Plastics
    "1303",  # Nan Ya Plastics
    "2002",  # China Steel
    "3711",  # ASE Technology Holding
    "3008",  # Largan Precision
    "2357",  # Asustek Computer
    "2379",  # Realtek Semiconductor
    "3034",  # Novatek Microelectronics
    "2395",  # Advantech
    "2345",  # Interface Inc.
    "2892",  # First Financial Holding
    "5880",  # Cooperative Financial
    "2880",  # Hua Nan Financial
    "2885",  # Yuanta Financial
    "2887",  # Taishin Financial
    "2890",  # SinoPac Financial
    "3045",  # Taiwan Mobile
    "4938",  # Pegatron
    "2912",  # President Chain Store
    "1216",  # Uni-President Enterprises
    "1101",  # Taiwan Cement
    "2603",  # Evergreen Marine
    "2609",  # Yang Ming Marine Transport
    "2615",  # Wan Hai Lines
    "2801",  # Chang Hwa Commercial Bank
    "2883",  # KGI Financial
    "2327",  # Yageo
    "2474",  # Can-Fite BioPharma (Catcher Technology)
    "3231",  # Wistron
    "2409",  # AU Optronics
    "2376",  # Gigabyte Technology
    "1326",  # Formosa Chemicals & Fibre
    "6669",  # Wiwynn
    "3661",  # World (Advanced Semiconductor)
    "2377",  # Micro-Star International
    "3037",  # Unimicron Technology
    "2059",  # King Yuan Electronics
    "1590",  # Airtac International
]

# Static name fallback map: code → 中文名稱.
#
# PRIMARY source for stock names is the live-fetched 證券名稱 column from
# BWIBBU_d / T86 responses (see marketlib/stocks.py _parse_bwibbu_all /
# _parse_t86_all and resolve_name).  This map is only used when live data is
# unavailable.  It should be periodically verified against the official 0050
# holdings list, as constituent names can change (e.g. mergers, rebrandings).
#
# Keys MUST remain a subset of TW50_CONSTITUENTS.  Any code whose name cannot
# be confirmed with high confidence is intentionally omitted — resolve_name()
# will fall back to the code string rather than display a wrong name.
TW50_NAMES: dict[str, str] = {
    "2330": "台積電",
    "2317": "鴻海",
    "2454": "聯發科",
    "2308": "台達電",
    "2412": "中華電",
    "2881": "富邦金",
    "2882": "國泰金",
    "2891": "中信金",
    "2303": "聯電",
    "2886": "兆豐金",
    "2884": "玉山金",
    "1301": "台塑",
    "1303": "南亞",
    "2002": "中鋼",
    "3711": "日月光投控",
    "2892": "第一金",
    "2880": "華南金",
    "2885": "元大金",
    "2887": "台新新光金",
    "2890": "永豐金",
    "2883": "凱基金",
    "2801": "彰銀",
    "5880": "合庫金",
    "2379": "瑞昱",
    "2357": "華碩",
    "3034": "聯詠",
    "2395": "研華",
    "2345": "智邦",
    "3231": "緯創",
    "2382": "廣達",
    "2603": "長榮",
    "2609": "陽明",
    "2615": "萬海",
    "1216": "統一",
    "1101": "台泥",
    "2912": "統一超",
    "4938": "和碩",
    "2327": "國巨",
    "2474": "可成",
    "3045": "台灣大",
    "2409": "友達",
    "2376": "技嘉",
    "1326": "台化",
    "6669": "緯穎",
    "3661": "世芯-KY",
    "2377": "微星",
    "3037": "欣興",
    "2059": "川湖",
    "1590": "亞德客-KY",
}
