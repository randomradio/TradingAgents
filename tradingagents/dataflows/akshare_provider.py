"""
AkShare data provider for China A-shares and Hong Kong stocks.

Provides stock price data, fundamental data, and news for Chinese markets
using the akshare library (https://akshare.akfamily.xyz/).
"""

import pandas as pd
from datetime import datetime, timedelta


def _safe_import_akshare():
    try:
        import akshare as ak
        return ak
    except ImportError:
        raise ImportError(
            "akshare is required for China/HK market data. "
            "Install it with: pip install akshare"
        )


# ---------------------------------------------------------------------------
# Core stock data
# ---------------------------------------------------------------------------

def get_china_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Fetch OHLCV data for a China A-share stock using akshare.

    Args:
        symbol: 6-digit A-share ticker (e.g. "600519")
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted string with OHLCV data.
    """
    ak = _safe_import_akshare()

    # akshare expects dates as YYYYMMDD
    start_fmt = start_date.replace("-", "")
    end_fmt = end_date.replace("-", "")

    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_fmt,
            end_date=end_fmt,
            adjust="qfq",  # forward-adjusted prices
        )
    except Exception as e:
        return f"Error fetching China stock data for {symbol}: {e}"

    if df is None or df.empty:
        return f"No data found for China A-share {symbol} between {start_date} and {end_date}"

    # Standardize column names
    col_map = {
        "日期": "Date",
        "开盘": "Open",
        "收盘": "Close",
        "最高": "High",
        "最低": "Low",
        "成交量": "Volume",
        "成交额": "Amount",
        "振幅": "Amplitude",
        "涨跌幅": "Change%",
        "涨跌额": "Change",
        "换手率": "Turnover%",
    }
    df = df.rename(columns=col_map)

    return df.to_string(index=False)


def get_hk_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Fetch OHLCV data for a Hong Kong stock using yfinance (via .HK suffix).

    Falls back to yfinance since akshare HK data can be limited.
    """
    try:
        import yfinance as yf
    except ImportError:
        return "yfinance is required for HK stock data"

    # Ensure .HK suffix
    if not symbol.upper().endswith(".HK"):
        symbol = f"{symbol}.HK"

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
    except Exception as e:
        return f"Error fetching HK stock data for {symbol}: {e}"

    if df is None or df.empty:
        return f"No data found for HK stock {symbol} between {start_date} and {end_date}"

    return df.to_string()


# ---------------------------------------------------------------------------
# Fundamental data
# ---------------------------------------------------------------------------

def get_china_fundamentals(ticker: str, curr_date: str) -> str:
    """Fetch fundamental data for a China A-share stock.

    Returns key financial metrics including PE, PB, market cap, etc.
    """
    ak = _safe_import_akshare()

    parts = []

    # Basic stock info / real-time quote for key metrics
    try:
        df_spot = ak.stock_zh_a_spot_em()
        row = df_spot[df_spot["代码"] == ticker]
        if not row.empty:
            r = row.iloc[0]
            parts.append(f"Stock: {r.get('名称', ticker)} ({ticker})")
            parts.append(f"Latest Price: ¥{r.get('最新价', 'N/A')}")
            parts.append(f"Change: {r.get('涨跌幅', 'N/A')}%")
            parts.append(f"Volume: {r.get('成交量', 'N/A')}")
            parts.append(f"Amount: ¥{r.get('成交额', 'N/A')}")
            parts.append(f"Market Cap: ¥{r.get('总市值', 'N/A')}")
            parts.append(f"Float Market Cap: ¥{r.get('流通市值', 'N/A')}")
            parts.append(f"PE Ratio (TTM): {r.get('市盈率-动态', 'N/A')}")
            parts.append(f"PB Ratio: {r.get('市净率', 'N/A')}")
            parts.append(f"52-week High: ¥{r.get('年初至今涨跌幅', 'N/A')}% YTD")
        else:
            parts.append(f"No spot data found for {ticker}")
    except Exception as e:
        parts.append(f"Error fetching spot data: {e}")

    # Financial indicators
    try:
        df_fi = ak.stock_financial_abstract_ths(symbol=ticker)
        if df_fi is not None and not df_fi.empty:
            parts.append("\n--- Financial Summary (latest available) ---")
            parts.append(df_fi.head(4).to_string(index=False))
    except Exception:
        pass  # Optional data, don't fail

    return "\n".join(parts) if parts else f"No fundamental data available for {ticker}"


def get_china_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """Fetch balance sheet data for a China A-share stock."""
    ak = _safe_import_akshare()
    try:
        df = ak.stock_balance_sheet_by_report_em(symbol=ticker)
        if df is not None and not df.empty:
            return df.head(8).to_string(index=False)
        return f"No balance sheet data for {ticker}"
    except Exception as e:
        return f"Error fetching balance sheet for {ticker}: {e}"


def get_china_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """Fetch cash flow data for a China A-share stock."""
    ak = _safe_import_akshare()
    try:
        df = ak.stock_cash_flow_sheet_by_report_em(symbol=ticker)
        if df is not None and not df.empty:
            return df.head(8).to_string(index=False)
        return f"No cash flow data for {ticker}"
    except Exception as e:
        return f"Error fetching cash flow for {ticker}: {e}"


def get_china_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """Fetch income statement data for a China A-share stock."""
    ak = _safe_import_akshare()
    try:
        df = ak.stock_profit_sheet_by_report_em(symbol=ticker)
        if df is not None and not df.empty:
            return df.head(8).to_string(index=False)
        return f"No income statement data for {ticker}"
    except Exception as e:
        return f"Error fetching income statement for {ticker}: {e}"


# ---------------------------------------------------------------------------
# News data
# ---------------------------------------------------------------------------

def get_china_stock_news(ticker: str, start_date: str, end_date: str) -> str:
    """Fetch news for a China A-share stock using akshare."""
    ak = _safe_import_akshare()
    try:
        df = ak.stock_news_em(symbol=ticker)
        if df is None or df.empty:
            return f"No news found for {ticker}"

        # Filter by date range if date columns exist
        articles = []
        for _, row in df.head(10).iterrows():
            title = row.get("新闻标题", row.get("title", ""))
            date_col = row.get("发布时间", row.get("publish_time", ""))
            source = row.get("文章来源", row.get("source", ""))
            articles.append(f"[{date_col}] {title} (Source: {source})")

        return "\n".join(articles) if articles else f"No news articles for {ticker}"
    except Exception as e:
        return f"Error fetching news for {ticker}: {e}"


# ---------------------------------------------------------------------------
# Technical indicators (reuse stockstats via yfinance data)
# ---------------------------------------------------------------------------

def get_china_indicators(symbol: str, indicator_name: str, start_date: str, end_date: str) -> str:
    """Fetch technical indicators for China A-share stock.

    Uses akshare for price data, then computes indicators via stockstats.
    """
    ak = _safe_import_akshare()

    start_fmt = start_date.replace("-", "")
    end_fmt = end_date.replace("-", "")

    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_fmt,
            end_date=end_fmt,
            adjust="qfq",
        )
    except Exception as e:
        return f"Error fetching data for indicator calculation: {e}"

    if df is None or df.empty:
        return f"No data for indicator calculation for {symbol}"

    # Rename to standard columns for stockstats
    col_map = {
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
    }
    df = df.rename(columns=col_map)

    try:
        from stockstats import StockDataFrame
        sdf = StockDataFrame.retype(df)
        indicator_data = sdf[indicator_name]
        result_df = pd.DataFrame({"date": df["date"], indicator_name: indicator_data})
        return result_df.tail(30).to_string(index=False)
    except Exception as e:
        return f"Error computing indicator '{indicator_name}' for {symbol}: {e}"
