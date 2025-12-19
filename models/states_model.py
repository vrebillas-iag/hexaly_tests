import hexaly.optimizer as hx
import pandas as pd
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# =========================
# 0. DATA / PARAMETERS
# =========================
engines = {
    "E1": {"initial_position": "DEJ", "remaining_life": 5000},
    "E2": {"initial_position": "DEJ", "remaining_life": 5000},
    "E3": {"initial_position": "DVE", "remaining_life": 5000},
    "E4": {"initial_position": "DVE", "remaining_life": 5000},
    "E5": {"initial_position": "POOL", "remaining_life": 5000},
    "E6": {"initial_position": "POOL", "remaining_life": 5000}
}

aircrafts    = {"DEJ": {}, "DVE": {}}
aircraft_ids = list(aircrafts.keys())
A            = len(aircraft_ids)

time_periods = list(range(65))

# Life / maintenance parameters
L_H_MAX  = 5000        # maximum life cap
L_H_RES  = 5000        # life restored after maintenance
LCC      = 150         # life consumption per period
T_SV     = 4           # maintenance duration
MIN_LH_TO_SV      = 150
MAX_SV_PER_MONTH = 2

# Costs
C_SV_BASE              = 1_000_000
C_INSTALLATION_REMOVAL = 10_000

# Logical parameters
MIN_STAY             = 3          # minimum stay after installation
L_REMOVAL_TRIGGER    = 5 * LCC    # life threshold required to remove an engine
L_US_MAX             = 150        # maximum life when in US state
M                    = 2 * L_H_MAX  # generic big-M

# =========================
# 1. HEXALY MODEL
# =========================
opt = hx.HexalyOptimizer()
m   = opt.model

# State coding
STATE_POOL  = 0
STATE_US    = A + 1
STATE_MAINT = A + 2      # maintenance state
aircraft_index = {a_id: idx + 1 for idx, a_id in enumerate(aircraft_ids)}

# =========================
# 2. VARIABLES
# =========================
state    = {}  # discrete engine state
L_vars   = {}  # remaining life
start_sv = {}  # maintenance start
end_sv   = {}  # maintenance end
install  = {}  # installation indicator
removal  = {}  # removal indicator

# --- State, life and start_sv for all periods
for e_id in engines:
    for t in time_periods:
        state[e_id, t]    = m.int(STATE_POOL, STATE_MAINT)
        L_vars[e_id, t]   = m.int(-100, L_H_MAX)
        start_sv[e_id, t] = m.bool()

# --- install / removal (t=0 excluded)
for e_id in engines:
    for t in time_periods:
        if t == 0:
            continue
        install[e_id, t] = m.bool()
        removal[e_id, t] = m.bool()

# --- end_sv: end of maintenance, linked with offset T_SV
for e_id in engines:
    for t in time_periods:
        if t >= T_SV:
            end_sv[e_id, t] = m.bool()
            m.constraint(end_sv[e_id, t] == start_sv[e_id, t - T_SV])
        else:
            end_sv[e_id, t] = 0  # not applicable yet

# =========================
# 3. INITIAL CONDITIONS (C1)
# =========================
for e_id, e_obj in engines.items():
    init_pos = e_obj["initial_position"]

    # Initial state: either POOL or installed in an aircraft
    if init_pos == "POOL":
        m.constraint(m.eq(state[e_id, 0], STATE_POOL))
    else:
        idx_a = aircraft_index[init_pos]
        m.constraint(m.eq(state[e_id, 0], idx_a))

    # Initial remaining life
    m.constraint(L_vars[e_id, 0] == e_obj["remaining_life"])

# =========================
# 4. 2 ENGINES PER AIRCRAFT PER PERIOD (C2)
# =========================
for t in time_periods:
    m.constraint(
        m.sum(m.eq(state[e_id, t], aircraft_index[a_id]) for e_id in engines for a_id in aircraft_ids) == 2 * A
    )

# =========================
# 5. LIFE EVOLUTION + MAINTENANCE RESET (C3)
# =========================
for e_id in engines:
    for k in range(len(time_periods) - 1):
        t      = time_periods[k]
        t_next = time_periods[k + 1]

        # flying[t] = 1 if engine is installed in an aircraft
        flying = m.sum(
            m.eq(state[e_id, t], aircraft_index[a_id]) for a_id in aircraft_ids
        )

        # End of maintenance at t_next
        end = end_sv[e_id, t_next] if t_next >= T_SV else 0

        # (C3.1) Life evolution when no maintenance finishes
        m.constraint(
            L_vars[e_id, t_next]
            >= L_vars[e_id, t] - LCC * flying - M * end
        )
        m.constraint(
            L_vars[e_id, t_next]
            <= L_vars[e_id, t] - LCC * flying + M * end
        )

        # (C3.2) Life reset when maintenance completes
        m.constraint(
            L_vars[e_id, t_next] >= L_H_RES - M * (1 - end)
        )
        m.constraint(
            L_vars[e_id, t_next] <= L_H_RES + M * (1 - end)
        )

# =========================
# 5.1 MINIMUM NUMBER OF SV PER MONTH (C4)
# =========================
for t in time_periods:
    m.constraint(
        m.sum(start_sv[e_id, t] for e_id in engines) <= MAX_SV_PER_MONTH
    )
# =========================
# 6. MINIMUM STAY IN AIRCRAFT AFTER INSTALLATION (C4)
# =========================
for e_id, e_obj in engines.items():
    init_pos      = e_obj["initial_position"]
    init_on_ac    = 1 if init_pos in aircraft_ids else 0

    # (C4.1) If engine starts installed, enforce minimum stay
    if init_on_ac == 1:
        for k in range(1, MIN_STAY + 1):
            if k in time_periods:
                m.constraint(m.eq(state[e_id, k], state[e_id, 0]))

    # (C4.2) If installed at time t, must stay MIN_STAY periods
    for t in time_periods:
        if t == 0:
            continue
        install_now = install[e_id, t]
        for k in range(1, MIN_STAY + 1):
            tk = t + k
            if tk in time_periods:
                m.constraint(
                    m.eq(state[e_id, tk], state[e_id, t]) >= install_now
                )

# =========================
# 7. MAINTENANCE LOGIC: COVERAGE, NO OVERLAPS, ORIGIN (C5–C7)
# =========================

# (C5) start_sv → T_SV consecutive MAINT periods
for e_id in engines:
    for t in time_periods:
        window = [tk for tk in range(t, t + T_SV) if tk in time_periods]
        if window:
            m.constraint(
                m.sum(m.eq(state[e_id, tk], STATE_MAINT) for tk in window)
                >= len(window) * start_sv[e_id, t]
            )

# (C6) No overlapping maintenance intervals for the same engine
for e_id in engines:
    for t in time_periods:
        window = [tau for tau in range(t, t + T_SV) if tau in time_periods]
        if window:
            m.constraint(
                m.sum(start_sv[e_id, tau] for tau in window) <= 1
            )

# (C7) MAINT allowed only if covered by a recent start_sv
for e_id in engines:
    for t in time_periods:
        window = [
            tau for tau in range(t - T_SV + 1, t + 1)
            if tau in time_periods
        ]
        if window:
            m.constraint(
                m.eq(state[e_id, t], STATE_MAINT)
                <= m.sum(start_sv[e_id, tau] for tau in window)
            )

# (C8) Maintenance can start only from AIRCRAFT or US
for e_id in engines:
    for t in time_periods:
        if t == 0:
            continue
        installed_prev = m.sum(
            m.eq(state[e_id, t - 1], aircraft_index[a_id])
            for a_id in aircraft_ids
        )
        was_us_prev = m.eq(state[e_id, t - 1], STATE_US)

        m.constraint(
            installed_prev + was_us_prev >= start_sv[e_id, t]
        )

# =========================
# 8. LIFE CONDITIONS IN POOL / US (C9)
# =========================
for e_id in engines:
    for t in time_periods:
        is_pool = m.eq(state[e_id, t], STATE_POOL)
        m.constraint(L_vars[e_id, t] >= 0 - M * (1 - is_pool))

        is_us = m.eq(state[e_id, t], STATE_US)
        m.constraint(L_vars[e_id, t] <= L_US_MAX + M * (1 - is_us))

# =========================
# 9. LOGICAL DEFINITIONS OF install / removal (C10, C11)
# =========================

for e_id in engines:
    for t in time_periods:
        if t == 0:
            continue

        # install = POOL(t-1) ∧ AIRCRAFT(t)
        was_pool_prev = m.eq(state[e_id, t - 1], STATE_POOL)
        on_ac_now = m.sum(
            m.eq(state[e_id, t], aircraft_index[a_id]) for a_id in aircraft_ids
        )

        m.constraint(install[e_id, t] <= was_pool_prev)
        m.constraint(install[e_id, t] <= on_ac_now)
        m.constraint(install[e_id, t] >= was_pool_prev + on_ac_now - 1)

        # removal = AIRCRAFT(t-1) ∧ ¬AIRCRAFT(t)
        on_ac_prev = m.sum(
            m.eq(state[e_id, t - 1], aircraft_index[a_id])
            for a_id in aircraft_ids
        )
        on_ac_now  = m.sum(
            m.eq(state[e_id, t], aircraft_index[a_id])
            for a_id in aircraft_ids
        )
        not_on_ac_now = 1 - on_ac_now

        m.constraint(removal[e_id, t] <= on_ac_prev)
        m.constraint(removal[e_id, t] <= not_on_ac_now)
        m.constraint(removal[e_id, t] >= on_ac_prev + not_on_ac_now - 1)

# =========================
# 10. FORBIDDEN TRANSITIONS (C12)
# =========================

# No direct jump from Aircraft A → Aircraft B
for e_id in engines:
    for t in time_periods:
        if t == 0:
            continue
        for a1 in aircraft_ids:
            for a2 in aircraft_ids:
                if a1 == a2:
                    continue
                idx1 = aircraft_index[a1]
                idx2 = aircraft_index[a2]
                m.constraint(
                    m.eq(state[e_id, t - 1], idx1)
                    + m.eq(state[e_id, t], idx2)
                    <= 1
                )

# No direct transition US(t-1) → Aircraft(t)
for e_id in engines:
    for t in time_periods:
        if t == 0:
            continue
        for a_id in aircraft_ids:
            idx_a = aircraft_index[a_id]
            m.constraint(
                m.eq(state[e_id, t - 1], STATE_US)
                + m.eq(state[e_id, t], idx_a)
                <= 1
            )

# =========================
# 11. REMOVAL CONDITIONS (C13)
# =========================

for e_id in engines:
    for t in time_periods:
        if t == 0:
            continue
        m.constraint(
            L_vars[e_id, t - 1]
            <= L_REMOVAL_TRIGGER + M * (1 - removal[e_id, t])
        )

# Limit total installations/removals per period
for t in time_periods:
    if t == 0:
        continue
    m.constraint(m.sum(install[e_id, t] for e_id in engines) <= 2)
    m.constraint(m.sum(removal[e_id, t] for e_id in engines) <= 2)

# =========================
# 12. LINE OF BALANCE (C14)
# =========================
# for t in time_periods:
#     m.constraint(
#         m.sum(m.eq(state[e_id, t], STATE_POOL) for e_id in engines) >= 1
#     )

# =========================
# 13. OBJECTIVE FUNCTION
# =========================
obj = 0.0
for e_id in engines:
    for t in time_periods:
        year_factor = 1.05 ** (t // 12)

        obj += C_SV_BASE * year_factor * start_sv[e_id, t]

        if t > 0:
            obj += C_INSTALLATION_REMOVAL * year_factor * install[e_id, t]
            obj += C_INSTALLATION_REMOVAL * year_factor * removal[e_id, t]

m.minimize(obj)

opt.param.time_limit = 1000
m.close()
opt.solve()

print("Status:", opt.solution.status)
for e_id in engines:
    print("Engine", e_id)
    for t in time_periods:
        s = state[e_id, t].value
        L = L_vars[e_id, t].value
        print(f"  t={t}: state={s}, L={L}")

print("Objective value:", obj.value)

for e_id in engines:
    for t in time_periods:
        sv_start = start_sv[e_id, t].value
        if sv_start > 0.5:
            print("Engine", e_id)
            print(f"  t={t}: SV started")

for e_id in engines:
    for t in time_periods:
        if start_sv[e_id, t].value > 0.5:
            print(f"SV en motor {e_id} empieza en t={t}")
            for k in range(T_SV):
                tk = t + k
                if tk in time_periods:
                    print("  ", tk, state[e_id, tk].value)

for e_id in engines:
    for t in time_periods:
        if t == 0:
            continue

        if install[e_id, t].value > 0.5:
            print(
                f"INSTALL {e_id} en t={t}: "
                f"state[t-1]={state[e_id, t-1].value}, state[t]={state[e_id, t].value}"
            )

        if removal[e_id, t].value > 0.5:
            print(
                f"REMOVAL {e_id} en t={t}: "
                f"state[t-1]={state[e_id, t-1].value}, state[t]={state[e_id, t].value}"
            )

# Imprimir la Line of balance
print("\nLine of Balance (engines in POOL over time):")
for t in time_periods:
    num_in_pool = sum(
        1 for e_id in engines
        if state[e_id, t].value == STATE_POOL
    )
    print(f"  t={t}: {num_in_pool} engines in POOL")


# Mapear los valores de state a etiquetas legibles
def state_label(val, aircraft_index):
    if val == 0:
        return "POOL"
    elif val == len(aircraft_index) + 1:
        return "US"
    elif val == len(aircraft_index) + 2:
        return "SV"
    else:
        # Buscar el id del avión correspondiente al índice
        for a_id, idx in aircraft_index.items():
            if idx == val:
                return f"AC_{a_id}"
        return f"AC_{val}"


# Añadir L al dataframe
data = []
for e_id in engines:
    for t in time_periods:
        s = state[e_id, t].value
        pos = state_label(s, aircraft_index)
        L = L_vars[e_id, t].value
        data.append({'Engine': e_id, 'Period': t, 'Position': pos, 'L': L})

df = pd.DataFrame(data)
engines_list = list(engines.keys())
fig = make_subplots(rows=len(engines_list), cols=1, shared_xaxes=True,
                    subplot_titles=[f'Engine {e_id}' for e_id in engines_list])

for i, e_id in enumerate(engines_list, start=1):
    df_e = df[df['Engine'] == e_id]
    fig.add_trace(
        go.Scatter(
            x=df_e['Period'],
            y=df_e['Position'],
            mode='lines+markers+text',
            name=f'Engine {e_id}',
            hovertext=[f"L={l:.0f}" for l in df_e['L']],
            hoverinfo='text'
        ),
        row=i, col=1
    )

fig.update_layout(height=200 * len(engines_list), title_text='Posición de motores en el tiempo (state) con L')
fig.update_yaxes(title_text='Posición')
fig.update_xaxes(title_text='Periodo')
fig.show()
output_dir = 'data/output_data'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'engine_positions.html')
fig.write_html(output_path)

