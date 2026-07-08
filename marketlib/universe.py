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
