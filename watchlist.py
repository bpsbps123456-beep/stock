PROFIT_SECTORS = {
    "빅테크":        ["AAPL", "MSFT", "AMZN", "META", "GOOGL"],
    "AI 반도체":     ["NVDA", "AVGO", "AMD", "MRVL", "CRDO", "ALAB"],
    "메모리":        ["MU", "SNDK", "WDC", "STX"],
    "네트워크/광통신": ["ANET", "CIEN", "LITE", "GLW"],
    "반도체 장비":   ["LRCX", "TER"],
    "엔터프라이즈 SW": ["ORCL"],
}

LOSS_SECTORS = {
    "AI 인프라":      ["CRWV", "NBIS", "IREN", "CIFR", "APLD"],
    "양자컴퓨팅":     ["IONQ", "RGTI"],
    "eVTOL":         ["JOBY"],
    "헬스케어 AI":    ["TEM"],
    "원자력/에너지":  ["UUUU", "SMR", "OKLO"],
    "반도체 장비(적자)": ["AEHR"],
    "반도체(적자)":   ["INTC", "NVTS"],
}

SECTOR_MAP = {**PROFIT_SECTORS, **LOSS_SECTORS}

TICKER_TO_SECTOR = {t: s for s, tickers in SECTOR_MAP.items() for t in tickers}

PROFIT_TICKERS = [t for tickers in PROFIT_SECTORS.values() for t in tickers]
LOSS_TICKERS   = [t for tickers in LOSS_SECTORS.values()  for t in tickers]
