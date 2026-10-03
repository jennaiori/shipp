import numpy as np
from shipp.components import Storage, Production
from shipp.timeseries import TimeSeries
from shipp.kernel_pyomo import solve_lp_pyomo_sizing, solve_lp_pyomo
import matplotlib.pyplot as plt
import warnings

VISUALIZE = False

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

def test_sizing_formulation_smoke_lp_alt():
    """Smoke test for formulation='lp_alt'."""
    n = 24
    dt = 1.0
    price = np.tile([50, 80, 60, 120], n // 4)
    prof_wind = np.abs(np.sin(np.arange(n) / 6.0))
    prof_pv = np.maximum(0, np.sin(np.arange(n) / 12.0 - 1.0))

    prod_wind = Production(TimeSeries(prof_wind, dt), p_cost=100000,
                           opex_fix=16000, p_max=None,
                           p_max_is_decision=True)
    prod_pv = Production(TimeSeries(prof_pv, dt), p_cost=80000,
                         opex_fix=12000, p_max=None,
                         p_max_is_decision=True)

    stor1 = Storage(e_cap=None, p_cap=None, eff_in=0.95, eff_out=0.95,
                    p_cost=150_000, e_cost=75_000, duration=4.0)
    stor2 = Storage(e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0,
                    p_cost=0, e_cost=0, duration=1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        os = solve_lp_pyomo_sizing(
            TimeSeries(price, dt),
            prod_wind, prod_pv,
            TimeSeries(prof_wind, dt), TimeSeries(prof_pv, dt),
            stor1, stor2,
            discount_rate=0.03, n_year=20,
            p_grid_max=500, n=n,
            options=dict(name_solver='gurobi', formulation='lp_alt'))

    assert os.production_list[0].p_max > 0
    assert os.production_list[1].p_max > 0
    assert abs(os.storage_list[0].e_cap
               - os.storage_list[0].p_cap * stor1.duration) < 1e-6
    assert hasattr(os, 'npv_objective')


def test_sizing_formulation_smoke_lp():
    """Smoke test for formulation='lp'."""
    n = 24
    dt = 1.0
    price = np.tile([50, 80, 60, 120], n // 4)
    prof_wind = np.abs(np.sin(np.arange(n) / 6.0))
    prof_pv = np.maximum(0, np.sin(np.arange(n) / 12.0 - 1.0))

    prod_wind = Production(TimeSeries(prof_wind, dt), p_cost=100000,
                           opex_fix=16000, p_max=None,
                           p_max_is_decision=True)
    prod_pv = Production(TimeSeries(prof_pv, dt), p_cost=80000,
                         opex_fix=12000, p_max=None,
                         p_max_is_decision=True)

    stor1 = Storage(e_cap=None, p_cap=None, eff_in=0.95, eff_out=0.95,
                    p_cost=150_000, e_cost=75_000, duration=4.0)
    stor2 = Storage(e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0,
                    p_cost=0, e_cost=0, duration=1.0)

    os = solve_lp_pyomo_sizing(
        TimeSeries(price, dt),
        prod_wind, prod_pv,
        TimeSeries(prof_wind, dt), TimeSeries(prof_pv, dt),
        stor1, stor2,
        discount_rate=0.03, n_year=20,
        p_grid_max=500, n=n,
        options=dict(name_solver='gurobi', formulation='lp'))

    assert os.production_list[0].p_max > 0
    assert os.production_list[1].p_max > 0
    assert abs(os.storage_list[0].e_cap
               - os.storage_list[0].p_cap * stor1.duration) < 1e-6
    assert hasattr(os, 'npv_objective')


def test_sizing_formulation_smoke_milp():
    """Smoke test for formulation='milp'."""
    n = 24
    dt = 1.0
    price = np.tile([50, 80, 60, 120], n // 4)
    prof_wind = np.abs(np.sin(np.arange(n) / 6.0))
    prof_pv = np.maximum(0, np.sin(np.arange(n) / 12.0 - 1.0))

    prod_wind = Production(TimeSeries(prof_wind, dt), p_cost=100000,
                           opex_fix=16000, p_max=None,
                           p_max_is_decision=True)
    prod_pv = Production(TimeSeries(prof_pv, dt), p_cost=80000,
                         opex_fix=12000, p_max=None,
                         p_max_is_decision=True)

    stor1 = Storage(e_cap=None, p_cap=None, eff_in=0.95, eff_out=0.95,
                    p_cost=150_000, e_cost=75_000, duration=4.0)
    stor2 = Storage(e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0,
                    p_cost=0, e_cost=0, duration=1.0)

    os = solve_lp_pyomo_sizing(
        TimeSeries(price, dt),
        prod_wind, prod_pv,
        TimeSeries(prof_wind, dt), TimeSeries(prof_pv, dt),
        stor1, stor2,
        discount_rate=0.03, n_year=20,
        p_grid_max=500, n=n,
        options=dict(name_solver='gurobi', formulation='milp'))

    assert os.check_losses(1e-4)
    assert os.production_list[0].p_max > 0
    assert os.production_list[1].p_max > 0
    assert abs(os.storage_list[0].e_cap
               - os.storage_list[0].p_cap * stor1.duration) < 1e-6
    assert hasattr(os, 'npv_objective')


def _reduction_setup():
    """Shared setup for the reduction tests."""
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

    prod_wind_legacy = Production(TimeSeries(prof_wind_abs, dt),
                                  p_cost=1_000_000, p_max=x_wind)
    prod_pv_legacy = Production(TimeSeries(prof_pv_unit, dt),
                                p_cost=1_000_000, p_max=0.0)
    stor_real_legacy = Storage(e_cap=stor_p_cap * stor_duration,
                               p_cap=stor_p_cap,
                               eff_in=0.95, eff_out=0.95,
                               p_cost=0.0, e_cost=0.0)
    stor_null_legacy = Storage(e_cap=0.0, p_cap=0.0,
                               eff_in=1.0, eff_out=1.0,
                               p_cost=0.0, e_cost=0.0)

    prof_wind_unit_ts = TimeSeries(prof_wind_unit, dt)
    prof_pv_unit_ts = TimeSeries(prof_pv_unit, dt)
    prod_wind_sizing = Production(TimeSeries(prof_wind_unit, dt),
                                  p_cost=1_000_000, opex_fix=0.0,
                                  p_max=x_wind)
    prod_pv_sizing = Production(TimeSeries(prof_pv_unit, dt),
                                p_cost=1_000_000, opex_fix=0.0,
                                p_max=0.0)
    stor_real_sizing = Storage(e_cap=None, p_cap=stor_p_cap,
                               eff_in=0.95, eff_out=0.95,
                               p_cost=0.0, e_cost=0.0,
                               duration=stor_duration)
    stor_null_sizing = Storage(e_cap=None, p_cap=0.0,
                               eff_in=1.0, eff_out=1.0,
                               p_cost=0.0, e_cost=0.0, duration=1.0)

    return dict(
        n=n, dt=dt, price_ts=price_ts,
        prod_wind_legacy=prod_wind_legacy,
        prod_pv_legacy=prod_pv_legacy,
        stor_real_legacy=stor_real_legacy,
        stor_null_legacy=stor_null_legacy,
        prod_wind_sizing=prod_wind_sizing,
        prod_pv_sizing=prod_pv_sizing,
        stor_real_sizing=stor_real_sizing,
        stor_null_sizing=stor_null_sizing,
        prof_wind_unit_ts=prof_wind_unit_ts,
        prof_pv_unit_ts=prof_pv_unit_ts,
        x_wind=x_wind,
        stor_p_cap=stor_p_cap,
        stor_duration=stor_duration,
        dp_min=dp_min,
        dp_max=dp_max,
    )


def _reduction_assertions(os_legacy, os_sizing, s):
    assert abs(os_sizing.production_list[0].p_max - s["x_wind"]) < 1e-6
    assert abs(os_sizing.production_list[1].p_max - 0.0) < 1e-6
    assert abs(os_sizing.storage_list[0].p_cap - s["stor_p_cap"]) < 1e-6
    assert abs(os_sizing.storage_list[0].e_cap
               - s["stor_p_cap"] * s["stor_duration"]) < 1e-6
    assert abs(os_sizing.storage_list[1].p_cap) < 1e-6

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

    for os_ in (os_legacy, os_sizing):
        ramp = np.diff(os_.power_out.data)
        assert np.all(ramp >= s["dp_min"] - 1e-4)
        assert np.all(ramp <= s["dp_max"] + 1e-4)


def test_sizing_reduces_to_legacy_lp_alt():
    s = _reduction_setup()
    os_legacy = solve_lp_pyomo(
        s["price_ts"], s["prod_wind_legacy"], s["prod_pv_legacy"],
        s["stor_real_legacy"], s["stor_null_legacy"],
        discount_rate=0.03, n_year=2,
        p_max=1e6, n=s["n"], p_min=0,
        dp_min=s["dp_min"], dp_max=s["dp_max"],
        options=dict(name_solver='gurobi', formulation='lp_alt'))

    os_sizing = solve_lp_pyomo_sizing(
        s["price_ts"], s["prod_wind_sizing"], s["prod_pv_sizing"],
        s["prof_wind_unit_ts"], s["prof_pv_unit_ts"],
        s["stor_real_sizing"], s["stor_null_sizing"],
        discount_rate=0.03, n_year=2,
        p_grid_max=1e6, n=s["n"], p_min=0,
        dp_min=s["dp_min"], dp_max=s["dp_max"],
        options=dict(name_solver='gurobi', formulation='lp_alt'))

    _reduction_assertions(os_legacy, os_sizing, s)

    if VISUALIZE:
        plot_schedule(os_legacy, s["price_ts"].data,
                      title="Reduction lp_alt - legacy kernel")
        plot_schedule(os_sizing, s["price_ts"].data,
                      title="Reduction lp_alt - sizing kernel")


def test_sizing_reduces_to_legacy_lp():
    s = _reduction_setup()
    os_legacy = solve_lp_pyomo(
        s["price_ts"], s["prod_wind_legacy"], s["prod_pv_legacy"],
        s["stor_real_legacy"], s["stor_null_legacy"],
        discount_rate=0.03, n_year=2,
        p_max=1e6, n=s["n"], p_min=0,
        dp_min=s["dp_min"], dp_max=s["dp_max"],
        options=dict(name_solver='gurobi', formulation='lp'))

    os_sizing = solve_lp_pyomo_sizing(
        s["price_ts"], s["prod_wind_sizing"], s["prod_pv_sizing"],
        s["prof_wind_unit_ts"], s["prof_pv_unit_ts"],
        s["stor_real_sizing"], s["stor_null_sizing"],
        discount_rate=0.03, n_year=2,
        p_grid_max=1e6, n=s["n"], p_min=0,
        dp_min=s["dp_min"], dp_max=s["dp_max"],
        options=dict(name_solver='gurobi', formulation='lp'))

    _reduction_assertions(os_legacy, os_sizing, s)

    if VISUALIZE:
        plot_schedule(os_legacy, s["price_ts"].data,
                      title="Reduction lp - legacy kernel")
        plot_schedule(os_sizing, s["price_ts"].data,
                      title="Reduction lp - sizing kernel")


def test_sizing_reduces_to_legacy_milp():
    s = _reduction_setup()
    os_legacy = solve_lp_pyomo(
        s["price_ts"], s["prod_wind_legacy"], s["prod_pv_legacy"],
        s["stor_real_legacy"], s["stor_null_legacy"],
        discount_rate=0.03, n_year=2,
        p_max=1e6, n=s["n"], p_min=0,
        dp_min=s["dp_min"], dp_max=s["dp_max"],
        options=dict(name_solver='gurobi', formulation='milp'))

    os_sizing = solve_lp_pyomo_sizing(
        s["price_ts"], s["prod_wind_sizing"], s["prod_pv_sizing"],
        s["prof_wind_unit_ts"], s["prof_pv_unit_ts"],
        s["stor_real_sizing"], s["stor_null_sizing"],
        discount_rate=0.03, n_year=2,
        p_grid_max=1e6, n=s["n"], p_min=0,
        dp_min=s["dp_min"], dp_max=s["dp_max"],
        options=dict(name_solver='gurobi', formulation='milp'))

    _reduction_assertions(os_legacy, os_sizing, s)

    if VISUALIZE:
        plot_schedule(os_legacy, s["price_ts"].data,
                      title="Reduction milp - legacy kernel")
        plot_schedule(os_sizing, s["price_ts"].data,
                      title="Reduction milp - sizing kernel")

def _fictitious_charge(os_, eff, dt):
    e = np.asarray(os_.storage_e[0].data, dtype=float)
    p = np.asarray(os_.storage_p[0].data, dtype=float)
    de = e[1:] - e[:-1]
    p_trim = p[:len(de)]
    slack_c = -(de + dt * eff * p_trim)
    slack_d = -(de + dt / eff * p_trim)
    both = (slack_c > 1e-6) & (slack_d > 1e-6)
    de_phys = (-dt * eff * np.where(p_trim < 0, p_trim, 0.0)
               - dt / eff * np.where(p_trim >= 0, p_trim, 0.0))
    return int(both.sum()), float(np.where(both,
                                           np.abs(de - de_phys),
                                           0.0).sum())


def test_sizing_kernel_fixed_production_lp_alt():
    _check_sizing_kernel_fixed_production('lp_alt')


def test_sizing_kernel_fixed_production_lp():
    _check_sizing_kernel_fixed_production('lp')


def test_sizing_kernel_fixed_production_milp():
    _check_sizing_kernel_fixed_production('milp')


def _check_sizing_kernel_fixed_production(formulation):
    """Does the sizing kernel reproduce degenerate storage model behavior
    when production capacity is fixed?

    For formulation='lp_alt' the expectation is that pinning production
    to the values the kernel itself chose removes the degeneracy. For
    'lp' and 'milp' the split formulations should be clean regardless,
    so the test only checks that the free-production run is not worse
    than the pinned-production run.
    """
    n = 24 * 7
    dt = 1.0

    hours = np.arange(n)
    trend = 50 + 70 * (hours / n)
    intraday = 25.0 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 5.0, size=n)
    price = np.clip(trend + intraday + noise, 1.0, None)
    price_ts = TimeSeries(price, dt)

    prof_unit = np.clip(
        0.5 + 0.5 * np.sin(2 * np.pi * hours / (24 * 3)), 0, 1)
    prof_unit_ts = TimeSeries(prof_unit, dt)
    pv_unit_ts = TimeSeries(np.zeros(n), dt)

    p_grid = 80.0
    eff = 0.95

    prod_wind_free = Production(prof_unit_ts, p_cost=1000000.0,
                                opex_fix=0.0, p_max=None,
                                p_max_is_decision=True)
    prod_pv_free = Production(pv_unit_ts, p_cost=1.0,
                              opex_fix=0.0, p_max=None,
                              p_max_is_decision=True)
    stor_free = Storage(e_cap=None, p_cap=None,
                        eff_in=eff, eff_out=eff,
                        p_cost=150_000.0, e_cost=150_000.0,
                        soc_min=0.0, soc_max=1.0, duration=2)
    stor_null = Storage(e_cap=None, p_cap=0.0,
                        eff_in=1.0, eff_out=1.0,
                        p_cost=0.0, e_cost=0.0, duration=1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        os_free = solve_lp_pyomo_sizing(
            price_ts=price_ts,
            prod1=prod_wind_free, prod2=prod_pv_free,
            prof1_unit=prof_unit_ts, prof2_unit=pv_unit_ts,
            stor1=stor_free, stor2=stor_null,
            discount_rate=0.03, n_year=20,
            p_grid_max=p_grid, n=n,
            options=dict(name_solver='gurobi',
                         formulation=formulation))

    x1 = os_free.production_list[0].p_max
    x2 = os_free.production_list[1].p_max

    prod_wind_pinned = Production(prof_unit_ts, p_cost=1000000.0,
                                  opex_fix=0.0, p_max=x1,
                                  p_max_is_decision=True)
    prod_pv_pinned = Production(pv_unit_ts, p_cost=1.0,
                                opex_fix=0.0, p_max=x2,
                                p_max_is_decision=True)
    stor_pinned = Storage(e_cap=None, p_cap=None,
                          eff_in=eff, eff_out=eff,
                          p_cost=150_000.0, e_cost=150_000.0,
                          soc_min=0.0, soc_max=1.0, duration=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        os_pinned = solve_lp_pyomo_sizing(
            price_ts=price_ts,
            prod1=prod_wind_pinned, prod2=prod_pv_pinned,
            prof1_unit=prof_unit_ts, prof2_unit=pv_unit_ts,
            stor1=stor_pinned, stor2=stor_null,
            discount_rate=0.03, n_year=20,
            p_grid_max=p_grid, n=n,
            options=dict(name_solver='gurobi',
                         formulation=formulation))

    n_free, fict_free = _fictitious_charge(os_free, eff, dt)
    n_pinned, fict_pinned = _fictitious_charge(os_pinned, eff, dt)

    msg = (
        f"[{formulation}] free prod   : {n_free} both-slack, "
        f"{fict_free:.1f} MWh fict\n"
        f"[{formulation}] pinned prod : {n_pinned} both-slack, "
        f"{fict_pinned:.1f} MWh fict"
    )
    print(f"\n  {msg}")

    if formulation == 'lp_alt':
        # The free-production run must exhibit the degeneracy
        assert fict_free > 1.0, (
            f"[{formulation}] free-production sizing kernel is clean "
            f"(fict = {fict_free:.1f} MWh); this test is only meaningful "
            f"when the degeneracy is present"
        )
        # Pinning production to the values the sizing kernel itself chose
        # must remove the degeneracy
        assert fict_pinned < 1.0, (
            f"[{formulation}] pinning production to x1={x1:.3f}, "
            f"x2={x2:.3f} does not remove the degeneracy: "
            f"{n_pinned} both-slack, {fict_pinned:.1f} MWh fict. "
            f"The production sizing degree of freedom is not necessary "
            f"for it."
        )
    else:
        # Split formulations should be clean regardless of whether
        # production is free or pinned
        assert fict_free < 1.0, (
            f"[{formulation}] free-production sizing kernel is not "
            f"physically valid: {n_free} both-slack, "
            f"{fict_free:.1f} MWh fict"
        )
        assert fict_pinned < 1.0, (
            f"[{formulation}] pinned-production sizing kernel is not "
            f"physically valid: {n_pinned} both-slack, "
            f"{fict_pinned:.1f} MWh fict"
        )

def test_sizing_free_capacities_across_storage_cost():
    """Compare the three sizing formulations across a range of storage
    p_cap costs with both production and storage capacity free.

    Comparable quantities across formulations are the capacity
    decisions, the objective value, and (for lp and milp) the grid
    dispatch. lp_alt is a relaxation and is not expected to match milp
    on power_out.
    """
    n = 24 * 30
    dt = 1.0

    hours = np.arange(n)
    trend = 50 + 70 * (hours / n)
    intraday = 25.0 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 5.0, size=n)
    price = np.clip(trend + intraday + noise, 1.0, None)
    price_ts = TimeSeries(price, dt)

    prof_unit = np.clip(
        0.5 + 0.5 * np.sin(2 * np.pi * hours / (24 * 3)), 0, 1)
    prof_unit_ts = TimeSeries(prof_unit, dt)
    pv_unit_ts = TimeSeries(np.zeros(n), dt)

    p_grid = 80.0
    eff = 0.95
    duration = 2.0

    p_costs = [1e2, 1e4, 1e5, 1.25e5, 1.5e5, 1.75e5, 2.0e5,
               2.25e5, 2.5e5, 2.75e5, 3e5, 1e6, 1e8]

    rows = []

    for p_cost in p_costs:
        row = {'p_cost': p_cost}
        for formulation in ('lp_alt', 'lp', 'milp'):
            prod_wind = Production(prof_unit_ts,
                                   p_cost=1_000_000.0, opex_fix=0.0,
                                   p_max=None, p_max_is_decision=True)
            prod_pv = Production(pv_unit_ts,
                                 p_cost=1.0, opex_fix=0.0,
                                 p_max=None, p_max_is_decision=True)
            stor = Storage(e_cap=None, p_cap=None,
                           eff_in=eff, eff_out=eff,
                           p_cost=p_cost, e_cost=p_cost,
                           soc_min=0.0, soc_max=1.0,
                           duration=duration)
            stor_null = Storage(e_cap=None, p_cap=0.0,
                                eff_in=1.0, eff_out=1.0,
                                p_cost=0.0, e_cost=0.0, duration=1.0)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                os_ = solve_lp_pyomo_sizing(
                    price_ts=price_ts,
                    prod1=prod_wind, prod2=prod_pv,
                    prof1_unit=prof_unit_ts, prof2_unit=pv_unit_ts,
                    stor1=stor, stor2=stor_null,
                    discount_rate=0.03, n_year=20,
                    p_grid_max=p_grid, n=n,
                    options=dict(name_solver='gurobi',
                                 formulation=formulation))

            row[(formulation, 'os')] = os_
            row[(formulation, 'x1')] = os_.production_list[0].p_max
            row[(formulation, 'x2')] = os_.production_list[1].p_max
            row[(formulation, 'p_cap')] = os_.storage_list[0].p_cap
            row[(formulation, 'e_cap')] = os_.storage_list[0].e_cap
            row[(formulation, 'npv')] = os_.npv_objective
            row[(formulation, 'power_out')] = np.asarray(
                os_.power_out.data, dtype=float)
            row[(formulation, 'loss_ok')] = os_.check_losses(1e-4)

        rows.append(row)

    # Print the sweep
    header = (f"  {'p_cost':>10} | {'x1 (milp/lp/alt)':>30} | "
              f"{'p_cap (milp/lp/alt)':>30} | "
              f"{'npv (milp/lp/alt)':>30} | "
              f"{'|dP| milp-lp':>12} | {'|dP| milp-alt':>14} | "
              f"{'loss ok (lp/alt)':>17}")
    print(f"\n{header}")
    for row in rows:
        x1s = "/".join(f"{row[(f, 'x1')]:.2f}"
                       for f in ('milp', 'lp', 'lp_alt'))
        pcs = "/".join(f"{row[(f, 'p_cap')]:.2f}"
                       for f in ('milp', 'lp', 'lp_alt'))
        npvs = "/".join(f"{row[(f, 'npv')]:.3f}"
                        for f in ('milp', 'lp', 'lp_alt'))
        dp_lp = float(np.max(np.abs(
            row[('lp', 'power_out')] - row[('milp', 'power_out')])))
        dp_alt = float(np.max(np.abs(
            row[('lp_alt', 'power_out')] - row[('milp', 'power_out')])))
        loss_ok = "/".join(
            "Y" if row[(f, 'loss_ok')] else "N"
            for f in ('lp', 'lp_alt'))
        print(f"  {row['p_cost']:>10.1e} | {x1s:>30} | "
              f"{pcs:>30} | {npvs:>30} | {dp_lp:>12.4f} | "
              f"{dp_alt:>14.4f} | {loss_ok:>17}")

    # milp is the physical reference and must pass the loss check at
    # every cost point
    for row in rows:
        assert row[('milp', 'loss_ok')], (
            f"milp loss check failed at p_cost={row['p_cost']:.1e}"
        )

    # lp with the default epsilon should be clean and match milp on
    # power_out
    for row in rows:
        dp_lp = float(np.max(np.abs(
            row[('lp', 'power_out')] - row[('milp', 'power_out')])))
        assert dp_lp < 1e-3, (
            f"lp and milp disagree on power_out at "
            f"p_cost={row['p_cost']:.1e}: max |dP| = {dp_lp:.4f}"
        )

    # At any cost, the three formulations should agree on whether to
    # build storage at all
    for row in rows:
        p_caps = [row[(f, 'p_cap')] for f in ('milp', 'lp', 'lp_alt')]
        zero_flags = [abs(p) < 1e-3 for p in p_caps]
        assert all(zero_flags) or not any(zero_flags), (
            f"formulations disagree on whether to build storage at "
            f"p_cost={row['p_cost']:.1e}: p_caps = {p_caps}"
        )

    # At very high cost, no formulation builds storage
    high = rows[-1]
    for formulation in ('lp_alt', 'lp', 'milp'):
        assert abs(high[(formulation, 'p_cap')]) < 1e-3, (
            f"[{formulation}] builds storage at p_cost=1e8"
        )

    # At very low cost, storage power capacity is large
    low = rows[0]
    for formulation in ('lp_alt', 'lp', 'milp'):
        assert low[(formulation, 'p_cap')] > 10.0, (
            f"[{formulation}] builds almost no storage at p_cost=1e2: "
            f"p_cap = {low[(formulation, 'p_cap')]:.3f}"
        )

    # Production sizing should be stable across formulations at every
    # cost point
    for row in rows:
        x1s = [row[(f, 'x1')] for f in ('milp', 'lp', 'lp_alt')]
        assert max(x1s) - min(x1s) < 5.0, (
            f"production sizing differs across formulations at "
            f"p_cost={row['p_cost']:.1e}: x1 = {x1s}"
        )