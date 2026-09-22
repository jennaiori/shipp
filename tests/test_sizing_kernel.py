import numpy as np
from shipp.components import Storage, Production
from shipp.timeseries import TimeSeries
from shipp.kernel_pyomo import solve_lp_pyomo_sizing, solve_lp_pyomo
import matplotlib.pyplot as plt

VISUALIZE = True

def plot_schedule(os, price, title=""):
    dt = os.power_out.dt
    n = len(os.power_out.data)
    t = np.arange(n) * dt

    prod = [p.data[:n] for p in os.production_p]
    stor = [s.data[:n] for s in os.storage_p]

    fig, ax = plt.subplots(figsize=(10, 4))

    # Positive stacking for production and discharge
    bottom_pos = np.zeros(n)
    for i, p in enumerate(prod):
        ax.bar(t, p, width=dt, bottom=bottom_pos,
               label=f"production {i+1}", alpha=0.75)
        bottom_pos = bottom_pos + p
    for i, s in enumerate(stor):
        pos = np.maximum(s, 0)
        ax.bar(t, pos, width=dt, bottom=bottom_pos,
               label=f"storage {i+1} discharge", alpha=0.75)
        bottom_pos = bottom_pos + pos

    # Negative stacking for storage charge
    bottom_neg = np.zeros(n)
    for i, s in enumerate(stor):
        neg = np.minimum(s, 0)
        ax.bar(t, neg, width=dt, bottom=bottom_neg,
               label=f"storage {i+1} charge", alpha=0.75)
        bottom_neg = bottom_neg + neg

    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Power [MW]")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # Price on twin axis
    ax2 = ax.twinx()
    ax2.plot(t, price[:n], color="grey", lw=1.2, alpha=0.8, label="price")
    ax2.set_ylabel("Price [EUR/MWh]", color="grey")
    ax2.tick_params(axis="y", colors="grey")

    fig.tight_layout()
    fname = title.replace(" ", "_").replace("—", "-").replace("/", "-")
    fig.savefig(f"plot_{fname}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

def test_sizing_kernel_smoke():
    """Does the kernel produce a physically consistent schedule?"""
    n = 24
    dt = 1.0
    price = np.tile([50, 80, 60, 120], n // 4)
    prof_wind = np.abs(np.sin(np.arange(n) / 6.0))
    prof_pv = np.maximum(0, np.sin(np.arange(n) / 12.0 - 1.0))

    prod_wind = Production(TimeSeries(prof_wind, dt), p_cost=1_000_000,
                           opex_fix=16000, p_max=None)
    prod_pv = Production(TimeSeries(prof_pv, dt), p_cost=800_000,
                         opex_fix=12000, p_max=None)

    stor1 = Storage(e_cap=None, p_cap=None, eff_in=1.0, eff_out=0.85,
                    p_cost=150_000, e_cost=75_000, duration=4.0)
    stor2 = Storage(e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0,
                    p_cost=0, e_cost=0, duration=1.0)

    os = solve_lp_pyomo_sizing(
        TimeSeries(price, dt),
        prod_wind, prod_pv,
        TimeSeries(prof_wind, dt), TimeSeries(prof_pv, dt),
        stor1, stor2,
        discount_rate=0.03, n_year=20,
        p_max=500, n=n,
        options=dict(name_solver='gurobi'))

    assert os.check_losses(1e-4)
    assert abs(os.storage_list[0].e_cap
               - os.storage_list[0].p_cap * stor1.duration) < 1e-6
    assert os.production_list[0].p_max > 0
    assert os.production_list[1].p_max > 0
    assert hasattr(os, 'npv_objective')

    if VISUALIZE:
        plot_schedule(os, price, title="Smoke test — free capacities")


def test_sizing_reduces_to_legacy_at_fixed_capacity():
    """Does the sizing kernel reduce to the legacy kernel when
    capacities are fixed?"""
    n = 12
    dt = 1.0
    price = np.array([40, 45, 50, 55, 60, 65,
                      70, 65, 60, 55, 50, 45], dtype=float)
    prof_wind_unit = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3,
                               0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    prof_pv_unit = np.zeros(n)
    x_wind = 50.0
    prof_wind_abs = prof_wind_unit * x_wind
    stor_p_cap = 20.0
    stor_duration = 4.0
    dp_min = -5.0
    dp_max = 5.0

    price_ts = TimeSeries(price, dt)

    # Legacy side: absolute profiles, fixed capacities
    prod_wind_legacy = Production(TimeSeries(prof_wind_abs, dt),
                                  p_cost=1_000_000, p_max=x_wind)
    prod_pv_legacy = Production(TimeSeries(prof_pv_unit, dt),
                                p_cost=1_000_000, p_max=0.0)
    stor_real_legacy = Storage(e_cap=stor_p_cap * stor_duration,
                               p_cap=stor_p_cap,
                               eff_in=1.0, eff_out=0.9,
                               p_cost=0.0, e_cost=0.0)
    stor_null_legacy = Storage(e_cap=0.0, p_cap=0.0,
                               eff_in=1.0, eff_out=1.0,
                               p_cost=0.0, e_cost=0.0)

    # Sizing side: per-unit profiles, fixed capacities via mode fields
    prof_wind_unit_ts = TimeSeries(prof_wind_unit, dt)
    prof_pv_unit_ts = TimeSeries(prof_pv_unit, dt)
    prod_wind_sizing = Production(TimeSeries(prof_wind_unit, dt),
                                  p_cost=1_000_000, opex_fix=0.0,
                                  p_max=x_wind)
    prod_pv_sizing = Production(TimeSeries(prof_pv_unit, dt),
                                p_cost=1_000_000, opex_fix=0.0,
                                p_max=0.0)
    stor_real_sizing = Storage(e_cap=None, p_cap=stor_p_cap,
                               eff_in=1.0, eff_out=0.9,
                               p_cost=0.0, e_cost=0.0,
                               duration=stor_duration)
    stor_null_sizing = Storage(e_cap=None, p_cap=0.0,
                               eff_in=1.0, eff_out=1.0,
                               p_cost=0.0, e_cost=0.0, duration=1.0)

    os_legacy = solve_lp_pyomo(
        price_ts, prod_wind_legacy, prod_pv_legacy,
        stor_real_legacy, stor_null_legacy,
        discount_rate=0.03, n_year=2,
        p_max=1e6, n=n, p_min=0,
        dp_min=dp_min, dp_max=dp_max,
        options=dict(name_solver='gurobi'))

    os_sizing = solve_lp_pyomo_sizing(
        price_ts, prod_wind_sizing, prod_pv_sizing,
        prof_wind_unit_ts, prof_pv_unit_ts,
        stor_real_sizing, stor_null_sizing,
        discount_rate=0.03, n_year=2,
        p_max=1e6, n=n, p_min=0,
        dp_min=dp_min, dp_max=dp_max,
        options=dict(name_solver='gurobi'))

    # Capacities pinned correctly
    assert abs(os_sizing.production_list[0].p_max - x_wind) < 1e-6
    assert abs(os_sizing.production_list[1].p_max - 0.0) < 1e-6
    assert abs(os_sizing.storage_list[0].p_cap - stor_p_cap) < 1e-6
    assert abs(os_sizing.storage_list[0].e_cap
               - stor_p_cap * stor_duration) < 1e-6
    assert abs(os_sizing.storage_list[1].p_cap) < 1e-6

    # Dispatch matches element-wise
    np.testing.assert_allclose(
        os_sizing.power_out.data, os_legacy.power_out.data,
        atol=1e-3, err_msg="Net grid power differs")
    np.testing.assert_allclose(
        os_sizing.storage_e[0].data, os_legacy.storage_e[0].data,
        atol=1e-3, err_msg="Storage SoC differs")
    np.testing.assert_allclose(
        os_sizing.storage_p[0].data, os_legacy.storage_p[0].data,
        atol=1e-4, err_msg="Storage 1 power differs")
    np.testing.assert_allclose(
        os_sizing.storage_p[1].data, os_legacy.storage_p[1].data,
        atol=1e-4, err_msg="Storage 2 power differs")
    np.testing.assert_allclose(
        os_sizing.p_curtail.data, os_legacy.p_curtail.data,
        atol=1e-4, err_msg="Curtailment differs")

    # Both kernels pass the physical loss check
    assert os_legacy.check_losses(1e-4), "legacy loss check failed"
    assert os_sizing.check_losses(1e-4), "sizing loss check failed"

    # Ramps are active and respected by both
    for os_ in (os_legacy, os_sizing):
        ramp = np.diff(os_.power_out.data)
        assert np.all(ramp >= dp_min - 1e-4)
        assert np.all(ramp <= dp_max + 1e-4)

    if VISUALIZE:
        plot_schedule(os_legacy, price_ts.data, title="Reduction — legacy kernel")
        plot_schedule(os_sizing, price_ts.data, title="Reduction — sizing kernel")