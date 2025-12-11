# sim_core_v1_1.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd
from scenarios import (
    BID_PRICE_LOWER_BOUND,
    BID_PRICE_UPPER_BOUND,
    PRICE_CAP_KRW_PER_KWH,
)

# -----------------------------
# 1) 정산 엔진 + 자료구조
# -----------------------------


@dataclass(order=True)
class Bid:
    price: float  # 원/kWh
    source_name: str = field(compare=False)
    capacity_kw: float = field(compare=False)


@dataclass
class PowerSource:
    name: str
    capacity_kw: float
    availability: float
    variable_cost: float  # 원/kWh
    is_renewable: bool = False


@dataclass
class SimulationResult:
    smp_krw_per_kwh: float
    my_dispatch_kw: float
    total_demand_kw: float
    total_supply_kw: float
    dispatch_details: Dict[str, float]
    lng_needed: bool


class MarketSimulator:
    """
    가격 오름차순(Merit-order) 정산으로 SMP·낙찰량 계산
    """

    def __init__(self, national_power_sources: List[PowerSource]):
        self.competitors = national_power_sources

    def _prepare_competitor_bids(self) -> List[Bid]:
        bids: List[Bid] = []
        for src in self.competitors:
            avail_cap = src.capacity_kw * src.availability
            if avail_cap > 0:
                bids.append(
                    Bid(
                        price=src.variable_cost,
                        source_name=src.name,
                        capacity_kw=avail_cap,
                    )
                )
        return bids

    def _market_clear(
        self, demand_kw: float, all_bids: List[Bid]
    ) -> Tuple[float, Dict[str, float], bool]:
        # 가격 오름차순 정렬
        sorted_bids = sorted(all_bids)
        cum_kw, smp = 0.0, 0.0
        dispatch = {b.source_name: 0.0 for b in sorted_bids}

        # LNG 제외 공급이 수요를 커버하는지 체크
        supply_wo_lng = sum(
            b.capacity_kw for b in sorted_bids if "LNG" not in b.source_name
        )
        lng_was_needed = supply_wo_lng < demand_kw

        if demand_kw <= 0 or not sorted_bids:
            return (
                (sorted_bids[0].price if sorted_bids else 0.0),
                dispatch,
                lng_was_needed,
            )

        for b in sorted_bids:
            if cum_kw < demand_kw:
                take = min(b.capacity_kw, demand_kw - cum_kw)
                dispatch[b.source_name] = take
                cum_kw += take
                smp = b.price
            else:
                break

        # 공급이 모자라면 SMP는 상한가로
        if cum_kw < demand_kw:
            smp = PRICE_CAP_KRW_PER_KWH

        return smp, dispatch, lng_was_needed

    def run(self, demand_kw: float, my_solar_bid: Bid) -> SimulationResult:
        all_bids = self._prepare_competitor_bids() + [my_solar_bid]
        smp, dispatch, lng_needed = self._market_clear(demand_kw, all_bids)
        my_dispatch = dispatch.get(my_solar_bid.source_name, 0.0)
        total_supply = sum(dispatch.values())
        return SimulationResult(
            smp_krw_per_kwh=smp,
            my_dispatch_kw=my_dispatch,
            total_demand_kw=demand_kw,
            total_supply_kw=total_supply,
            dispatch_details=dispatch,
            lng_needed=lng_needed,
        )


# -----------------------------
# 3) 경쟁자 구성 (시나리오×시점)
# -----------------------------


def create_competitor_power_sources(
    market_condition: pd.Series, scenario_config: dict
) -> List[PowerSource]:
    """
    시점별 시장 상태(market_condition)와 시나리오 설정(scenario_config)을 받아
    경쟁 발전원 리스트를 구성.
    """
    sources: List[PowerSource] = []
    pconf = scenario_config["power_sources"]
    mix = scenario_config["competitor_strategy_mix"]

    # 화석/원전/양수/연료전지 등
    for name, at in pconf.items():
        if at["capacity_kw"] > 0:
            cost = at["cost"]
            sources.append(
                PowerSource(
                    name=name,
                    capacity_kw=at["capacity_kw"],
                    availability=at["availability"],
                    variable_cost=cost,
                    is_renewable=(cost <= BID_PRICE_UPPER_BOUND),
                )
            )

    # 태양광(전략 분할)
    total_solar = float(market_condition.get("competitor_solar_energy", 0.0))
    for strat, ratio in mix.get("solar", {}).items():
        if ratio > 0 and total_solar > 0:
            if strat == "AGGRESSIVE":
                price = BID_PRICE_LOWER_BOUND
            elif strat == "NEUTRAL":
                price = (BID_PRICE_LOWER_BOUND + BID_PRICE_UPPER_BOUND) / 2
            else:
                price = BID_PRICE_UPPER_BOUND
            sources.append(
                PowerSource(
                    name=f"경쟁-태양광_{strat}",
                    capacity_kw=total_solar * ratio,
                    availability=1.0,
                    variable_cost=price,
                    is_renewable=True,
                )
            )

    # 풍력(전략 분할)
    total_wind = float(market_condition.get("wind_energy", 0.0))
    for strat, ratio in mix.get("wind", {}).items():
        if ratio > 0 and total_wind > 0:
            if strat == "AGGRESSIVE":
                price = BID_PRICE_LOWER_BOUND
            elif strat == "NEUTRAL":
                price = (BID_PRICE_LOWER_BOUND + BID_PRICE_UPPER_BOUND) / 2
            else:
                price = BID_PRICE_UPPER_BOUND
            sources.append(
                PowerSource(
                    name=f"경쟁-풍력_{strat}",
                    capacity_kw=total_wind * ratio,
                    availability=1.0,
                    variable_cost=price,
                    is_renewable=True,
                )
            )

    return sources


# -----------------------------
# 4) CSV 기반 데이터 파이프라인 (15분)
# -----------------------------
#CSV_PATH = "version1_data.csv" # 원본 코드 주석 line 1줄, 경로 변경
CSV_PATH = "./Project/Henergy/Demo_scheme_dev01_(sim_core)/version1_data.csv" #추가 코드 by KH
HEGY_SHARE = 0.01  # 태양광 중 우리 VPP 몫 (1차 버전: 1%)
DEFAULT_OPER_RESERVE_RATE = 10.0  # 운영 예비율 고정값 (1차 버전)


def prepare_simulation_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """
    version1_data.csv 기준으로 RL 환경에서 사용할 데이터프레임 생성.

    input CSV columns:
        - present_load
        - renew_power_total
        - renew_power_solar
        - renew_power_wind
        - da
        - renew_power_solar_scaled
        - renew_power_wind_scaled
        - renew_power_total_scaled

    output columns:
        - datetime
        - forecast_load
        - hegy_solar_energy
        - competitor_solar_energy
        - wind_energy
        - other_renew_energy  # (총 재생 - 태양 - 풍력)
        - oper_reserve_rate
        - da (가격 정보 그대로 유지)
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # index를 datetime 컬럼으로 노출
    df = df.rename_axis("datetime").reset_index()

    # 1차 버전: forecast_load = present_load
    df["forecast_load"] = df["present_load"].astype(float)

    # 태양광/풍력은 스케일된 값 사용
    solar = df["renew_power_solar_scaled"].astype(float)
    wind = df["renew_power_wind_scaled"].astype(float)

    # 총 재생에너지
    total_renew = df["renew_power_total_scaled"].astype(float)

    # 우리 VPP 몫 / 경쟁자 몫
    df["hegy_solar_energy"] = solar * HEGY_SHARE
    df["competitor_solar_energy"] = solar - df["hegy_solar_energy"]

    # 풍력은 통째로 경쟁자側으로 간주 (1차 버전)
    df["wind_energy"] = wind

    # 기타 재생에너지 = 총 재생 - 태양 - 풍력 (음수 방지)
    other = (total_renew - solar - wind).clip(lower=0.0)
    df["other_renew_energy"] = other

    # 예비율은 고정값 (추후 실제값으로 바꾸면 됨)
    df["oper_reserve_rate"] = DEFAULT_OPER_RESERVE_RATE

    return df
