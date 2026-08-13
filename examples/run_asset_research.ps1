# AetherDrift Atlas research command catalog
#
# Run from the AetherDrift-Atlas directory. Each basket is research only and
# writes separate artifacts to outputs/. Do not combine results after looking at
# them; compare the predeclared baskets and their research_gate decisions.

$atlasPython = ".\.venv\Scripts\python.exe"
$common = @("--start", "2018-01-01", "--trees", "300")

# 1. Broad US equity / ETF basket: establishes a liquid equity baseline.
& $atlasPython -m aetherdrift_atlas `
  --symbols AAPL MSFT JPM XLE XLV SPY QQQ `
  --asset-classes equity equity equity etf etf etf etf `
  @common `
  --output outputs\research_us_equities

# 2. Technology concentration: evaluate separately from broad equities.
& $atlasPython -m aetherdrift_atlas `
  --symbols AAPL MSFT NVDA AMD QQQ `
  --asset-classes equity equity equity equity etf `
  @common `
  --output outputs\research_technology

# 3. Defensive sectors: a distinct regime hypothesis, not a diversification claim.
& $atlasPython -m aetherdrift_atlas `
  --symbols XLP XLV XLU IEF TLT `
  --asset-classes etf etf etf etf etf `
  @common `
  --output outputs\research_defensive

# 4. Major cryptocurrencies: use separate results because crypto trades 24/7.
& $atlasPython -m aetherdrift_atlas `
  --symbols BTC-USD ETH-USD SOL-USD `
  --asset-classes crypto crypto crypto `
  @common `
  --output outputs\research_crypto

# 5. Major FX pairs: inspect carefully because Yahoo Finance volume may be limited.
& $atlasPython -m aetherdrift_atlas `
  --symbols EURUSD=X GBPUSD=X USDJPY=X AUDUSD=X `
  --asset-classes forex forex forex forex `
  @common `
  --output outputs\research_forex

# 6. Commodity proxies: ETFs avoid futures-roll assumptions in this first pass.
& $atlasPython -m aetherdrift_atlas `
  --symbols GLD SLV USO DBA `
  --asset-classes commodity commodity commodity commodity `
  @common `
  --output outputs\research_commodities

# 7. Cross-asset diversification candidate. Compare only after each sleeve is
# evaluated independently; this is not a recommendation to allocate capital.
& $atlasPython -m aetherdrift_atlas `
  --symbols SPY TLT GLD BTC-USD EURUSD=X `
  --asset-classes etf etf commodity crypto forex `
  @common `
  --output outputs\research_cross_asset
