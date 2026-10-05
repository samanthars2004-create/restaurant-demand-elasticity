"""
Step 1 - Download raw data from FRED into data/raw/fredgraph.csv.

Monthly series (all from FRED, Federal Reserve Bank of St. Louis):
  RSFSDP         Retail sales: food services & drinking places ($ millions, SA), Census
  CUSR0000SEFV   CPI: food away from home (menu prices, SA), BLS
  CUSR0000SAF11  CPI: food at home (grocery prices, SA), BLS
  CPIAUCSL       CPI: all items (SA), BLS
  DSPIC96        Real disposable personal income (bil. chained 2017 $), BEA
  CES7000000008  Avg hourly earnings, production & nonsupervisory employees,
                 leisure & hospitality ($/hour, SA), BLS
  WPU02          PPI: processed foods and feeds (NSA), BLS

If the download fails (e.g., no internet), open FRED_URL in a browser, save the
file as data/raw/fredgraph.csv and rerun the pipeline with --no-download.
"""
from pathlib import Path
import urllib.request

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FRED_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
    "RSFSDP,CUSR0000SEFV,CUSR0000SAF11,CPIAUCSL,DSPIC96,CES7000000008,WPU02"
)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / "fredgraph.csv"
    try:
        req = urllib.request.Request(FRED_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
        print("downloaded fredgraph.csv")
    except Exception as exc:
        if dest.exists():
            print(f"could not download ({exc}); using existing copy")
        else:
            raise SystemExit(f"Download failed ({exc}). Save {FRED_URL} as {dest}")


if __name__ == "__main__":
    main()
