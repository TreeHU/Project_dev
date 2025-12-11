# --- 시뮬레이션 공통 설정 ---
REC_PRICE_KRW_PER_KWH: float = 68.6
BID_PRICE_LOWER_BOUND: float = REC_PRICE_KRW_PER_KWH * -2.5
BID_PRICE_UPPER_BOUND: float = 0
PRICE_CAP_KRW_PER_KWH: float = 260.0

# ==============================================================================
# 시나리오 1: 제10차 전력수급기본계획 기반 (2030년)
# - 현재 정부의 공식 계획을 반영한 가장 현실적인 기준 시나리오입니다.
# - 석탄을 LNG로 점진적 전환, 원전은 현상 유지, 재생에너지는 꾸준히 확대됩니다.
# ==============================================================================
SCENARIO_1_GOV_PLAN = {
    "description": "시나리오 1: 제10차 전력수급기본계획 기반 (2030년)",
    "power_sources": {
        # 발전원: {설비용량(kW), 가동률, 변동비(원/kWh)}
        "원자력": {
            "capacity_kw": 26550000 / 4,
            "availability": 0.85,
            "cost": 82.0,
        },
        "유연탄": {
            "capacity_kw": 32940000 / 4,
            "availability": 0.50,
            "cost": 160.5,
        },  # 용량 감소
        "무연탄": {
            "capacity_kw": 400000 / 4,
            "availability": 0.50,
            "cost": 140.0,
        },
        "유류": {
            "capacity_kw": 531000 / 4,
            "availability": 0.05,
            "cost": 360.0,
        },
        "LNG": {
            "capacity_kw": 58315000 / 4,
            "availability": 0.35,
            "cost": 165.8,
        },  # 용량 증가 (석탄 대체)
        "양수": {
            "capacity_kw": 6500000 / 4,
            "availability": 0.10,
            "cost": 200.0,
        },  # 용량 증가
        "연료전지": {
            "capacity_kw": 2800000 / 4,
            "availability": 0.90,
            "cost": BID_PRICE_UPPER_BOUND,
        },
        # 기타 신재생은 태양광/풍력에 통합하여 단순화
    },
    "renewable_capacity": {  # 별도 관리하여 Multiplier 적용
        "solar": 25000000,
        "wind": 5000000,
    },
    "competitor_strategy_mix": {
        "solar": {"AGGRESSIVE": 0.6, "NEUTRAL": 0.3, "CONSERVATIVE": 0.1},
        "wind": {"AGGRESSIVE": 0.6, "NEUTRAL": 0.3, "CONSERVATIVE": 0.1},
    },
}

# ==============================================================================
# 시나리오 2: 원자력 중심의 무탄소에너지(CFE) 전환 (2035년)
# - 차기 정부 정책이 원자력 확대로 전환되는 경우를 가정한 시나리오입니다.
# - 신규 원전 및 SMR(소형모듈원전) 도입으로 원자력 비중이 크게 늘고, 강력한 탄소 규제로 석탄/LNG 비용이 급등합니다.
# ==============================================================================
SCENARIO_2_NUCLEAR_CFE = {
    "description": "시나리오 2: 원자력 중심의 무탄소에너지(CFE) 전환 (2035년)",
    "power_sources": {
        "원자력": {
            "capacity_kw": 35000000 / 4,
            "availability": 0.90,
            "cost": 88.0,
        },  # 대형 원전 및 SMR 추가
        "유연탄": {
            "capacity_kw": 15000000 / 4,
            "availability": 0.20,
            "cost": 320.0,
        },  # 탄소세 등으로 비용 급등
        "무연탄": {"capacity_kw": 0, "availability": 0, "cost": 0},
        "유류": {"capacity_kw": 0, "availability": 0, "cost": 0},
        "LNG": {
            "capacity_kw": 40000000 / 4,
            "availability": 0.20,
            "cost": 350.0,
        },  # 백업 역할로 축소, 비용 급등
        "양수": {
            "capacity_kw": 7000000 / 4,
            "availability": 0.15,
            "cost": 210.0,
        },
        "연료전지": {
            "capacity_kw": 3500000 / 4,
            "availability": 0.90,
            "cost": BID_PRICE_UPPER_BOUND,
        },
    },
    "renewable_capacity": {
        "solar": 35000000,
        "wind": 8000000,
    },
    "competitor_strategy_mix": {  # 원자력과 경쟁해야 하므로 공격적 입찰 비중 증가
        "solar": {"AGGRESSIVE": 0.8, "NEUTRAL": 0.2, "CONSERVATIVE": 0.0},
        "wind": {"AGGRESSIVE": 0.8, "NEUTRAL": 0.2, "CONSERVATIVE": 0.0},
    },
}

# ==============================================================================
# 시나리오 3: 재생에너지 중심의 급진적 전환 (2035년)
# - RE100 이행 등 기업과 사회의 요구로 재생에너지 보급이 폭발적으로 증가하는 시나리오입니다.
# - 태양광/풍력 설비가 막대하게 늘고, 유연성 자원(ESS, 양수)의 역할이 중요해집니다. 원전은 점진적으로 감축됩니다.
# ==============================================================================
SCENARIO_3_RENEWABLE_LEAD = {
    "description": "시나리오 3: 재생에너지 중심의 급진적 전환 (2035년)",
    "power_sources": {
        "원자력": {
            "capacity_kw": 20000000 / 4,
            "availability": 0.80,
            "cost": 95.0,
        },  # 노후 원전 폐쇄로 용량 감소
        "유연탄": {
            "capacity_kw": 5000000 / 4,
            "availability": 0.10,
            "cost": 380.0,
        },  # 거의 퇴출
        "무연탄": {"capacity_kw": 0, "availability": 0, "cost": 0},
        "유류": {"capacity_kw": 0, "availability": 0, "cost": 0},
        "LNG": {
            "capacity_kw": 30000000 / 4,
            "availability": 0.15,
            "cost": 400.0,
        },  # 계통 안정성 위한 최소 역할
        "양수": {
            "capacity_kw": 8000000 / 4,
            "availability": 0.20,
            "cost": 220.0,
        },  # 유연성 자원 확대
        "연료전지": {
            "capacity_kw": 4000000 / 4,
            "availability": 0.90,
            "cost": BID_PRICE_UPPER_BOUND,
        },
    },
    "renewable_capacity": {
        "solar": 80000000,  # 대규모 보급
        "wind": 20000000,  # 대규모 보급
    },
    "competitor_strategy_mix": {  # 공급 과잉으로 출력제어 회피 위한 경쟁 심화
        "solar": {"AGGRESSIVE": 0.9, "NEUTRAL": 0.1, "CONSERVATIVE": 0.0},
        "wind": {"AGGRESSIVE": 0.9, "NEUTRAL": 0.1, "CONSERVATIVE": 0.0},
    },
}

# ==============================================================================
# 시나리오 4: 에너지 위기 및 고비용 구조 고착화 (2030년)
# - 국제 정세 불안으로 LNG, 유연탄 등 원자재 가격이 급등하고 높은 수준을 유지하는 비관적 시나리오입니다.
# - 에너지 전환 투자가 위축되고, 높은 SMP가 지속됩니다.
# ==============================================================================
SCENARIO_4_ENERGY_CRISIS = {
    "description": "시나리오 4: 에너지 위기 및 고비용 구조 고착화 (2030년)",
    "power_sources": {
        "원자력": {
            "capacity_kw": 26550000 / 4,
            "availability": 0.80,
            "cost": 85.0,
        },  # 정비 등으로 가동률 저하
        "유연탄": {
            "capacity_kw": 35000000 / 4,
            "availability": 0.60,
            "cost": 280.0,
        },  # 고비용에도 가동률 상승
        "무연탄": {
            "capacity_kw": 400000 / 4,
            "availability": 0.60,
            "cost": 250.0,
        },
        "유류": {
            "capacity_kw": 531000 / 4,
            "availability": 0.10,
            "cost": 500.0,
        },  # 최고가 발전원
        "LNG": {
            "capacity_kw": 55000000 / 4,
            "availability": 0.40,
            "cost": 310.0,
        },  # 매우 비싸지만 필수 자원
        "양수": {
            "capacity_kw": 6500000 / 4,
            "availability": 0.10,
            "cost": 240.0,
        },
        "연료전지": {
            "capacity_kw": 2500000 / 4,
            "availability": 0.90,
            "cost": BID_PRICE_UPPER_BOUND,
        },
    },
    "renewable_capacity": {  # 투자 위축으로 보급 더딤
        "solar": 20000000,
        "wind": 4000000,
    },
    "competitor_strategy_mix": {  # SMP가 높아 굳이 공격적으로 입찰할 유인 감소
        "solar": {"AGGRESSIVE": 0.2, "NEUTRAL": 0.6, "CONSERVATIVE": 0.2},
        "wind": {"AGGRESSIVE": 0.2, "NEUTRAL": 0.6, "CONSERVATIVE": 0.2},
    },
}

# ==============================================================================
# 시나리오 5: 수소 경제의 부상 및 LNG 대체 (2040년)
# - 기술 발전으로 그린/블루수소 도입이 본격화되어 LNG의 역할을 대체하는 장기 시나리오입니다.
# - 수소 혼소/전소 발전이 새로운 중간 부하 발전원으로 등장합니다.
# ==============================================================================
SCENARIO_5_HYDROGEN_ECONOMY = {
    "description": "시나리오 5: 수소 경제의 부상 및 LNG 대체 (2040년)",
    "power_sources": {
        "원자력": {
            "capacity_kw": 30000000 / 4,
            "availability": 0.90,
            "cost": 90.0,
        },
        "유연탄": {"capacity_kw": 0, "availability": 0, "cost": 0},  # 완전 퇴출
        "무연탄": {"capacity_kw": 0, "availability": 0, "cost": 0},
        "유류": {"capacity_kw": 0, "availability": 0, "cost": 0},
        "LNG": {
            "capacity_kw": 10000000 / 4,
            "availability": 0.10,
            "cost": 380.0,
        },  # 잔존 물량
        "수소/암모니아": {
            "capacity_kw": 35000000 / 4,
            "availability": 0.40,
            "cost": 250.0,
        },  # 신규 도입, LNG 역할 대체
        "양수": {
            "capacity_kw": 8000000 / 4,
            "availability": 0.20,
            "cost": 230.0,
        },
        "연료전지": {
            "capacity_kw": 5000000 / 4,
            "availability": 0.90,
            "cost": BID_PRICE_UPPER_BOUND,
        },
    },
    "renewable_capacity": {
        "solar": 90000000,
        "wind": 30000000,
    },
    "competitor_strategy_mix": {
        "solar": {"AGGRESSIVE": 0.7, "NEUTRAL": 0.3, "CONSERVATIVE": 0.0},
        "wind": {"AGGRESSIVE": 0.7, "NEUTRAL": 0.3, "CONSERVATIVE": 0.0},
    },
}
