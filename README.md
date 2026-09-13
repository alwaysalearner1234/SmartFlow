# SmartFlow | Smart Order Routing & Risk-Aware Trade Execution

A production-quality quantitative trading system that models high-frequency market microstructure, predicts short-term adverse-selection risk with machine learning (Logistic Regression baseline and XGBoost), couples predictions with closed-form Almgren–Chriss execution-cost trajectories, dynamically orchestrates execution strategies (Market, TWAP, VWAP, Passive Maker, Aggressive Taker, AC Slicing, and Proposed Risk-Aware SOR), simulates fills under realistic market physics, and performs rigorous, identical-market backtesting visualized through an institutional Streamlit terminal.

---

## 1. Problem Statement

Executing large institutional orders in electronic limit order books (LOBs) entails a fundamental trade-off:
- **Aggressive Execution (Taker):** Immediately crosses the bid-ask spread to guarantee completion, but incurs substantial spread cost and adverse temporary/permanent market impact.
- **Passive Execution (Maker):** Posts resting limit orders at or inside the quote to earn the spread and avoid impact, but suffers from **adverse selection (the "winner's curse")**—resting orders are filled disproportionately when informed traders sweep the book right before an adverse price plunge.

Smart Order Routing (SOR) and risk-aware execution must dynamically sense toxic flow, estimate instantaneous execution costs, and adapt slicing and order placement in real time.

---

## 2. Why Smart Order Routing Matters

In fragmented and fast-moving modern markets:
- Static algorithms (e.g., fixed TWAP) blindly execute on arbitrary clocks regardless of whether a volatility spike or toxic flow is occurring.
- Naive liquidity-seeking strategies get picked off by predatory algorithms.
- **SmartFlow** bridges microstructure predictive signals (Order Flow Imbalance, micro-price divergence, spread expansion) with mathematical liquidation models (Almgren-Chriss) to minimize **Implementation Shortfall**.

---

## 3. System Architecture

```
Market Data (L2/L3 Order Book & Trades)
  │
  ▼
Data Cleaning & Validation (data/validator.py)
  │
  ▼
Order Book Representation (data/contracts.py)
  │
  ▼
Feature Engineering Pipeline (features/)
  ├── Order Book Depth & Micro-Price (features/order_book.py)
  ├── Order Flow Imbalance / OFI (features/imbalance.py)
  ├── Bid-Ask Spread Dynamics (features/spread.py)
  ├── Price Momentum & Velocity (features/momentum.py)
  ├── Realized Volatility Regimes (features/volatility.py)
  └── Trade Flow & CVD (features/trade_flow.py)
  │
  ▼
ML Adverse Selection Risk Model (models/)
  ├── Preprocessing & Zero-Leakage Splits (models/preprocessing.py)
  ├── Training (Logistic Regression Baseline vs. XGBoost) (models/train.py)
  └── Predictor Interface with Heuristic Fallback (models/predictor.py)
  │
  ▼
Dynamic Strategy Engine (execution/strategy_engine.py)
  ├── Risk Probability (ML Predictor)
  ├── Almgren–Chriss Cost/Variance Model (execution/almgren_chriss.py)
  ├── Current Market State (Spread, OFI, Depth, Volatility)
  ├── Execution Urgency & Time Progress
  └── Remaining Order Quantity
  │
  ▼
Strategy Selection & Execution (strategies/ & execution/)
  ├── Market (strategies/market.py)
  ├── TWAP (strategies/twap.py)
  ├── VWAP (strategies/vwap.py)
  ├── Passive Maker with Adverse Backoff (execution/passive.py)
  ├── Aggressive Spread Crossing (execution/aggressive.py)
  ├── Almgren-Chriss Trajectory (strategies/almgren_chriss_strat.py)
  └── Proposed Risk-Aware SOR (strategies/proposed.py)
  │
  ▼
Market & Fill Simulator (simulator/)
  ├── Chronological Market Replay (simulator/market_simulator.py)
  ├── Order Management & Lifecycle State Machine (execution/order_manager.py)
  ├── Fill Physics: Queue Decay & Adverse Bias (simulator/fill_model.py)
  └── Order Book Depletion & Price Impact (simulator/order_book_simulator.py)
  │
  ▼
Backtesting & Performance Analytics (backtest/)
  ├── Implementation Shortfall & Cost Metrics (backtest/metrics.py)
  ├── Multi-Strategy Fair Replay (backtest/backtester.py)
  └── 10-Scenario Stress Testing (backtest/comparison.py)
  │
  ▼
Refinitiv/Bloomberg Dark Terminal Dashboard (dashboard/app.py)
```

---

## 4. Market Microstructure Features

All features are deterministic and calculated without lookahead bias:
- **Order Book Depth & Micro-Price:** Level 1 to 5 depth, size-weighted Stoikov micro-price:
  $$P_{micro} = \frac{P_{bid} Q_{ask} + P_{ask} Q_{bid}}{Q_{bid} + Q_{ask}}$$
- **Order Flow Imbalance (OFI):** Cont-Kukanov-Stoikov formula tracking net changes in bid/ask price and depth:
  $$OFI = \Delta W_{bid} - \Delta W_{ask}$$
- **Spread Metrics:** Absolute spread ($P_{ask} - P_{bid}$), relative spread, spread in basis points (bps), and rolling spread expansion ratio.
- **Price Momentum:** Log-returns over 5, 10, 20, 50 ticks, price velocity, and acceleration.
- **Realized Volatility:** Rolling return standard deviation over 10, 30, 60 ticks and volatility regime ratios.
- **Trade Flow & CVD:** Buyer- vs. seller-initiated volume, volume imbalance, and Cumulative Volume Delta (CVD).

---

## 5. Machine Learning Adverse-Selection Model

### Horizon & Labeling Logic
- **Prediction Horizon ($H$):** Configurable (default: 10 ticks).
- **Target Label ($Y=1$):**
  - **For BUY Orders:** Adverse event occurs if future mid-price drops significantly below current mid-price:
    $$P_{t+H} \le P_t - \theta \cdot \text{Spread}_t$$
  - **For SELL Orders:** Adverse event occurs if future mid-price rises significantly above current mid-price:
    $$P_{t+H} \ge P_t + \theta \cdot \text{Spread}_t$$
- **Leakage Prevention:** Strictly chronological time-series splits (Train: 70%, Validation: 15%, Test: 15%). Scalers are fit strictly on training data.
- **Models Evaluated:**
  1. Regularized Logistic Regression Baseline
  2. XGBoost Classifier with probability calibration
- **Standard Interface:** `predict(features) -> PredictionResult` returning probability, timestamp, horizon, model name, and status (`TRAINED` or `FALLBACK_HEURISTIC`).

---

## 6. Almgren–Chriss Optimal Execution Model

Implements the classical Almgren-Chriss (2000) optimal execution framework:
- **Parameters:** Total quantity $X$, execution horizon $T$, slices $N$, risk aversion $\lambda$, temporary impact $\eta$, permanent impact $\gamma$, volatility $\sigma$.
- **Trajectory:**
  $$x_j = \frac{\sinh(\kappa (T - t_j))}{\sinh(\kappa T)} X_0, \quad \text{where } \kappa \approx \sqrt{\frac{\lambda \sigma^2}{\eta}}$$
- Computes:
  - Trajectory $x_j$ and slice quantities $v_j$
  - Expected execution cost $E[x] = \frac{1}{2} \gamma X^2 + \eta \sum \frac{v_j^2}{\tau}$
  - Cost variance $V[x] = \sigma^2 \tau \sum x_j^2$
  - Utility cost $U[x] = E[x] + \lambda V[x]$
  - Dynamic trajectory re-slicing as remaining time and inventory progress.

---

## 7. Dynamic Strategy Engine

At every decision interval, the engine synthesizes:
1. **Market State:** Spread, OFI, liquidity, and volatility
2. **ML Risk Score:** Probability of adverse selection ($p$)
3. **Almgren–Chriss Guidance:** Optimal slice size and theoretical cost
4. **Execution Urgency:** Progress of time elapsed vs. remaining inventory
5. **Remaining Quantity:** Order inventory left to liquidate

### Conceptual Decision Tree:
- **Near Deadline ($Urgency \ge 0.75$):** Aggressive liquidity sweep to prevent shortfall.
- **High Adverse Risk ($p \ge 0.65$):**
  - If high inventory remaining: Cross spread aggressively before adverse price jump occurs.
  - If low inventory remaining: Withdraw passive quotes to avoid being picked off (winner's curse).
- **Low Adverse Risk ($p \le 0.35$) & Tight Spread:** Post passive limit orders at best quote to capture spread.
- **Moderate Risk:** Follow Almgren-Chriss scheduled trajectory.

Every decision returns an auditable reason string explaining **WHY** the strategy was chosen.

---

## 8. Simulator & Fill Model

- **Deterministic Market Replay:** Ensures all strategies run across the **exact same tick sequence**.
- **Market Orders:** Depth-walking model that consumes multi-level order-book liquidity with realistic slippage and price impact.
- **Limit Orders:** Simulates queue position, priority decay, and adverse selection fill bias (higher probability of fills during toxic runs).
- **Endogenous Market Impact:** Order book simulator adjusts visible depth and shifts mid-prices post-execution.
- **Latency Delay:** Enforces realistic network and exchange matching latency (default: 15ms).

---

## 9. Backtesting & Performance Metrics

Metrics calculated from actual executions (no fabricated values):
- **Implementation Shortfall (IS):** Dollar and basis points vs. arrival price:
  $$IS_{bps} = \frac{(\bar{P}_{exec} - P_{arrival}) \cdot Q_{exec}}{P_{arrival} \cdot Q_{exec}} \times 10^4$$
- **Realized Execution Cost:** Total slippage and half-spread paid
- **Average Execution Price vs. Arrival Price**
- **Completion Rate & Fill Rate**
- **Adverse Selection Cost**
- **Almgren-Chriss Expected vs. Realized Cost**

### 10 Experimental Regimes:
1. Normal Market
2. High Volatility
3. Poor Liquidity / Thin Book
4. High Adverse-Selection Risk
5. Low Adverse-Selection Risk
6. Small Order
7. Medium Order
8. Large Order
9. Short Execution Horizon
10. Long Execution Horizon

---

## 10. Installation & Setup

### Prerequisites
- Python 3.9+ or 3.10+
- `pip`

```bash
git clone https://github.com/your-repo/smart-order-routing.git
cd smart-order-routing
pip install -r requirements.txt
```

---

## 11. Usage

### 1. Generate Synthetic Market Datasets
```bash
python main.py --generate-data
```
Generates 10 market regime datasets under `data/raw/` and `data/processed/`.

### 2. Train ML Adverse-Selection Models
```bash
python main.py --train
```
Extracts features, trains Logistic Regression and XGBoost models chronologically, outputs genuine classification metrics, and saves the best model to `models/saved/best_adverse_selection_model.joblib`.

### 3. Run Live Execution Demo
```bash
python main.py --run
```
Runs a complete end-to-end demonstration showing live order-book processing, feature extraction, ML risk scoring, AC scheduling, dynamic strategy shifts, order fills, and execution metrics.

### 4. Run Multi-Strategy Backtesting
```bash
python main.py --backtest
```
Runs Market, TWAP, VWAP, Almgren-Chriss, and Proposed strategies across all 10 experimental scenarios under identical market conditions and prints the comparative summary table.

### 5. Launch Quantitative Trading Terminal Dashboard
```bash
streamlit run dashboard/app.py
```
Or via CLI:
```bash
python main.py --dashboard
```

---

## 12. Testing

Execute the comprehensive pytest suite:
```bash
pytest tests/ -v
```
Tests cover:
- Microstructure features (micro-price, imbalance, OFI, spread bps, momentum, volatility, trade flow)
- ML model training, preprocessing, zero-leakage splits, predictor interface, and fallback handling
- Almgren-Chriss schedules, order manager lifecycle, passive/aggressive strategies, and dynamic engine
- Market simulator replay, queue position, fill physics, and endogenous price impact
- Multi-strategy backtester and identical-market-condition guarantees.

---

## 13. Limitations & Future Improvements

- **Real-Market Calibration:** In production, $\eta$ (temporary impact) and $\gamma$ (permanent impact) should be calibrated against institutional trade execution data (FIX / ITCH feeds).
- **Multi-Venue SOR:** The current system models single-venue multi-level books; future extensions can route across multi-exchange fragmented liquidity pools.
- **Deep Reinforcement Learning:** Coupling the dynamic strategy engine with Proximal Policy Optimization (PPO) or Deep Q-Networks (DQN) for continuous control.

---

## 14. License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 15. React Frontend (Phase 1)

An alternative React + Plotly frontend skeleton and its proposed data contract live in [frontend/](frontend/README.md). The Streamlit dashboard above is the currently integrated UI. The React frontend requires a FastAPI endpoint before it can display backend data; see its README for the outstanding integration decisions.
