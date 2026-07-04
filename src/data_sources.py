"""
Data acquisition layer for the Oil & Gas Market Analysis Dashboard.

VERIFIED DATA SOURCES
---------------------
1. FRED - Federal Reserve Bank of St. Louis (public, no key required for CSV download)
   - WTI Crude (Cushing, OK)         : series DCOILWTICO
   - Brent Crude (Europe)            : series DCOILBRENTEU
   - Henry Hub Natural Gas Spot      : series DHHNGSP
   https://fred.stlouisfed.org/

2. Yahoo Finance (via yfinance) - real-time / historical futures & energy equities
   - CL=F WTI futures, BZ=F Brent futures, NG=F Henry Hub gas futures
   - XLE Energy Select Sector ETF, USO United States Oil Fund
   https://finance.yahoo.com/

3. Geopolitical Risk (GPR) Index - Caldara & Iacoviello (Federal Reserve Board)
   The peer-reviewed standard for measuring geopolitical risk, used by central banks.
   Dauphin, T. et al.; Caldara, D., & Iacoviello, M. (2022), American Economic Review.
   https://www.matteoiacoviello.com/gpr.htm

4. U.S. Energy Information Administration (EIA) API v2 - OPTIONAL (free API key)
   Official U.S. government energy statistics (production, inventories, prices).
   https://www.eia.gov/opendata/

All network fetches are cached locally in ./data and degrade gracefully to a
clearly-labelled synthetic fallback so the dashboard always runs offline.
"""

from __future__ import annotations

import io
import os
import datetime as dt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CACHE_TTL_HOURS = 12
_HEADERS = {"User-Agent": "OilGasDashboard/1.0 (research/educational use)"}

# FRED series identifiers ----------------------------------------------------
FRED_SERIES = {
    "WTI": "DCOILWTICO",        # WTI spot, $/bbl
    "Brent": "DCOILBRENTEU",    # Brent spot, $/bbl
    "NatGas": "DHHNGSP",        # Henry Hub spot, $/MMBtu
}

# Yahoo Finance tickers ------------------------------------------------------
YF_TICKERS = {
    "WTI": "CL=F",
    "Brent": "BZ=F",
    "NatGas": "NG=F",
    "EnergyETF": "XLE",
    "OilFund": "USO",
}

# Official GPR daily index spreadsheet
GPR_DAILY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"


# --------------------------------------------------------------------------- #
# Cache helpers
# --------------------------------------------------------------------------- #
def _cache_path(name: str) -> Path:
    return DATA_DIR / f"{name}.parquet"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
    return age < dt.timedelta(hours=CACHE_TTL_HOURS)


def _read_cache(name: str) -> Optional[pd.DataFrame]:
    path = _cache_path(name)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


def _write_cache(name: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_cache_path(name))
    except Exception:
        # parquet engine missing -> fall back to csv silently
        try:
            df.to_csv(DATA_DIR / f"{name}.csv")
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# FRED (primary price source, no API key needed)
# --------------------------------------------------------------------------- #
def fetch_fred_series(series_id: str) -> Optional[pd.Series]:
    """Download a single FRED series as a daily pd.Series indexed by date."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
        s = pd.to_numeric(df[series_id], errors="coerce").dropna()
        s.name = series_id
        return s
    except Exception:
        return None


def fetch_prices_fred() -> Optional[pd.DataFrame]:
    cols = {}
    for label, sid in FRED_SERIES.items():
        s = fetch_fred_series(sid)
        if s is not None and len(s) > 0:
            cols[label] = s
    if not cols:
        return None
    df = pd.DataFrame(cols).sort_index()
    df = df.ffill().dropna(how="all")
    return df


# --------------------------------------------------------------------------- #
# Yahoo Finance (secondary / equities)
# --------------------------------------------------------------------------- #
def fetch_prices_yfinance(period: str = "10y") -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        tickers = list(YF_TICKERS.values())
        raw = yf.download(
            tickers, period=period, interval="1d",
            auto_adjust=True, progress=False, group_by="ticker",
        )
        if raw is None or len(raw) == 0:
            return None
        out = {}
        for label, tk in YF_TICKERS.items():
            try:
                out[label] = raw[tk]["Close"]
            except Exception:
                continue
        if not out:
            return None
        df = pd.DataFrame(out).sort_index()
        df.index = pd.to_datetime(df.index)
        return df.ffill().dropna(how="all")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Geopolitical Risk Index (Caldara & Iacoviello)
# --------------------------------------------------------------------------- #
def fetch_gpr_index() -> Optional[pd.DataFrame]:
    """Daily Geopolitical Risk index (GPRD) and threat/act sub-indices."""
    try:
        resp = requests.get(GPR_DAILY_URL, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content), engine="xlrd")
        df.columns = [str(c).strip().upper() for c in df.columns]

        # The file ships both a proper 'DATE' column and an integer 'DAY'
        # (yyyymmdd). Prefer DATE; fall back to parsing DAY as %Y%m%d.
        if "DATE" in df.columns:
            df["__dt"] = pd.to_datetime(df["DATE"], errors="coerce")
        elif "DAY" in df.columns:
            df["__dt"] = pd.to_datetime(
                df["DAY"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
            )
        else:
            df["__dt"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        df = df.dropna(subset=["__dt"]).set_index("__dt").sort_index()
        df.index.name = "date"

        rename = {}
        for c in df.columns:
            if c == "GPRD":
                rename[c] = "GPR"
            elif c == "GPRD_THREAT":
                rename[c] = "GPR_Threat"
            elif c == "GPRD_ACT":
                rename[c] = "GPR_Act"
        keep = [c for c in df.columns if c in rename]
        if not keep:
            num = df.select_dtypes("number")
            if num.empty:
                return None
            out = num.iloc[:, :1].rename(columns={num.columns[0]: "GPR"})
            return out
        out = df[keep].rename(columns=rename)
        return out.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# EIA API v2 (optional, official US gov source)
# --------------------------------------------------------------------------- #
def fetch_eia_series(route: str, api_key: Optional[str] = None) -> Optional[pd.Series]:
    api_key = api_key or os.environ.get("EIA_API_KEY")
    if not api_key:
        return None
    url = f"https://api.eia.gov/v2/{route}"
    params = {
        "api_key": api_key,
        "frequency": "daily",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        rows = resp.json()["response"]["data"]
        df = pd.DataFrame(rows)
        df["period"] = pd.to_datetime(df["period"], errors="coerce")
        s = pd.Series(
            pd.to_numeric(df["value"], errors="coerce").values,
            index=df["period"],
        ).dropna()
        return s
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Synthetic fallback (clearly labelled) so dashboard runs fully offline
# --------------------------------------------------------------------------- #
def _synthetic_prices(years: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    end = pd.Timestamp.today().normalize()
    idx = pd.bdate_range(end=end, periods=years * 252)
    n = len(idx)

    def gbm(start, mu, sigma):
        shocks = rng.normal(mu / 252, sigma / np.sqrt(252), n)
        return start * np.exp(np.cumsum(shocks))

    wti = gbm(70, 0.02, 0.35)
    brent = wti * (1.0 + 0.06 + rng.normal(0, 0.01, n))
    gas = gbm(3.0, 0.0, 0.55)
    df = pd.DataFrame(
        {
            "WTI": wti,
            "Brent": brent,
            "NatGas": gas,
            "EnergyETF": gbm(80, 0.05, 0.25),
            "OilFund": gbm(70, 0.02, 0.33),
        },
        index=idx,
    )
    return df.round(3)


def _synthetic_gpr(index: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    base = 100 + rng.normal(0, 25, len(index)).cumsum() * 0.0
    noise = np.abs(rng.normal(100, 35, len(index)))
    # add a few risk spikes
    spikes = np.zeros(len(index))
    for _ in range(6):
        p = rng.integers(0, len(index))
        spikes[p : p + 20] += rng.uniform(80, 200)
    gpr = np.clip(noise + spikes, 20, None)
    df = pd.DataFrame({"GPR": gpr}, index=index)
    df["GPR_Threat"] = gpr * rng.uniform(0.4, 0.6, len(index))
    df["GPR_Act"] = gpr - df["GPR_Threat"]
    return df.round(2)


# --------------------------------------------------------------------------- #
# Public orchestration
# --------------------------------------------------------------------------- #
def load_prices(force_refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Return (prices_df, source_label)."""
    if not force_refresh and _is_fresh(_cache_path("prices")):
        cached = _read_cache("prices")
        if cached is not None:
            return cached, "cache"

    df = fetch_prices_fred()
    source = "FRED (St. Louis Fed)"
    if df is None or df.shape[1] < 2:
        yf_df = fetch_prices_yfinance()
        if yf_df is not None:
            df = yf_df
            source = "Yahoo Finance"

    if df is None or df.empty:
        return _synthetic_prices(), "SYNTHETIC (offline demo data)"

    _write_cache("prices", df)
    return df, source


def load_gpr(force_refresh: bool = False) -> tuple[pd.DataFrame, str]:
    if not force_refresh and _is_fresh(_cache_path("gpr")):
        cached = _read_cache("gpr")
        if cached is not None:
            return cached, "cache"

    df = fetch_gpr_index()
    if df is None or df.empty:
        return None, "unavailable"

    _write_cache("gpr", df)
    return df, "Caldara & Iacoviello GPR (Fed Board)"


def load_all(force_refresh: bool = False) -> dict:
    prices, price_src = load_prices(force_refresh)
    gpr, gpr_src = load_gpr(force_refresh)

    if gpr is None:
        gpr = _synthetic_gpr(prices.index)
        gpr_src = "SYNTHETIC (GPR source unreachable)"

    return {
        "prices": prices,
        "prices_source": price_src,
        "gpr": gpr,
        "gpr_source": gpr_src,
    }


if __name__ == "__main__":
    bundle = load_all()
    print("Prices source:", bundle["prices_source"])
    print(bundle["prices"].tail())
    print("\nGPR source:", bundle["gpr_source"])
    print(bundle["gpr"].tail())
