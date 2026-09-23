import numpy as np
from shipp.components import Storage, Production
from shipp.timeseries import TimeSeries
from shipp.kernel_pyomo import solve_lp_pyomo_sizing, solve_lp_pyomo
import matplotlib.pyplot as plt
import warnings

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
                               eff_in=0.95, eff_out=0.95,
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
                               eff_in=0.95, eff_out=0.95,
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
        p_grid_max=1e6, n=n, p_min=0,
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


def test_sizing_kernel_fixed_production():
    """Does the sizing kernel reproduce degenerate storage model behavior
    when production capacity is fixed?
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

    # Free production, free storage: reproduces the degeneracy
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
            options=dict(name_solver='gurobi'))

    x1 = os_free.production_list[0].p_max
    x2 = os_free.production_list[1].p_max

    # Free storage, production pinned to the values chosen above
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
            options=dict(name_solver='gurobi'))

    def _fictitious(os_):
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

    n_free, fict_free = _fictitious(os_free)
    n_pinned, fict_pinned = _fictitious(os_pinned)

    print(f"\n  free prod   : {n_free} both-slack, {fict_free:.1f} MWh fict")
    print(f"  pinned prod : {n_pinned} both-slack, {fict_pinned:.1f} MWh fict")

    # The free-production run must exhibit the degeneracy
    assert fict_free > 1.0, (
        f"free-production sizing kernel is clean (fict = {fict_free:.1f} MWh); "
        f"this test is only meaningful when the degeneracy is present"
    )

    # Pinning production to the values the sizing kernel itself chose must
    # remove the degeneracy
    assert fict_pinned < 1.0, (
        f"pinning production to x1={x1:.3f}, x2={x2:.3f} does not remove "
        f"the degeneracy: {n_pinned} both-slack, {fict_pinned:.1f} MWh fict. "
        f"The production sizing degree of freedom is not necessary for it."
    )


def test_sizing_kernel_dispatch_behavior():
    """Does the sizing kernel produce a physically valid dispatch when
    production and storage capacities are both free?
    """
    n = 24 * 7          # one week
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

    # Step 1: sizing kernel, free production and free storage
    prod_wind_sz = Production(prof_unit_ts, p_cost=1000000.0,
                              opex_fix=0.0, p_max=None,
                              p_max_is_decision=True)
    prod_pv_sz = Production(pv_unit_ts, p_cost=1.0,
                            opex_fix=0.0, p_max=None,
                            p_max_is_decision=True)
    stor_sz = Storage(e_cap=None, p_cap=None,
                      eff_in=eff, eff_out=eff,
                      p_cost=150_000.0, e_cost=150_000.0,
                      soc_min=0.0, soc_max=1.0, duration=2)
    stor_sz_null = Storage(e_cap=None, p_cap=0.0,
                           eff_in=1.0, eff_out=1.0,
                           p_cost=0.0, e_cost=0.0, duration=1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        os_sz = solve_lp_pyomo_sizing(
            price_ts=price_ts,
            prod1=prod_wind_sz, prod2=prod_pv_sz,
            prof1_unit=prof_unit_ts, prof2_unit=pv_unit_ts,
            stor1=stor_sz, stor2=stor_sz_null,
            discount_rate=0.03, n_year=20,
            p_grid_max=p_grid, n=n,
            i_max=None, i_min=None,
            options=dict(name_solver='gurobi'))

    x1 = os_sz.production_list[0].p_max
    x2 = os_sz.production_list[1].p_max
    p_cap_sz = os_sz.storage_list[0].p_cap
    e_cap_sz = os_sz.storage_list[0].e_cap

    print(f"\n  === chain diagnostics, n = {n} ===")
    print(f"  [1] sizing, free prod + free storage")
    print(f"      x1* / x2*        : {x1:.3f} / {x2:.3f} MW")
    print(f"      p_cap* / e_cap*  : {p_cap_sz:.3f} / {e_cap_sz:.3f}")

    # Step 2: legacy kernel, production fixed, storage free
    wind_abs_ts = TimeSeries(prof_unit * x1, dt)
    pv_abs_ts = TimeSeries(np.zeros(n), dt)

    prod_wind_leg = Production(wind_abs_ts, p_cost=100_000.0,
                               p_max=x1)
    prod_pv_leg = Production(pv_abs_ts, p_cost=1.0, p_max=x2)

    stor_leg_free = Storage(e_cap=None, p_cap=None,
                            eff_in=eff, eff_out=eff,
                            p_cost=150_000.0, e_cost=150_000.0,
                            soc_min=0.0, soc_max=1.0)
    stor_leg_null = Storage(e_cap=0.0, p_cap=0.0,
                            eff_in=1.0, eff_out=1.0,
                            p_cost=0.0, e_cost=0.0)

    os_leg_free = solve_lp_pyomo(
        price_ts=price_ts,
        prod1=prod_wind_leg, prod2=prod_pv_leg,
        stor1=stor_leg_free, stor2=stor_leg_null,
        discount_rate=0.03, n_year=20,
        p_max=p_grid, n=n,
        options=dict(name_solver='gurobi'))

    p_cap_leg = os_leg_free.storage_list[0].p_cap
    e_cap_leg = os_leg_free.storage_list[0].e_cap

    print(f"  [2] legacy, fixed prod + free storage")
    print(f"      p_cap* / e_cap*  : {p_cap_leg:.3f} / {e_cap_leg:.3f}")

    # Step 3: legacy kernel, production fixed, storage fixed
    stor_leg_fixed = Storage(e_cap=e_cap_leg, p_cap=p_cap_leg,
                             eff_in=eff, eff_out=eff,
                             p_cost=0.0, e_cost=0.0,
                             soc_min=0.0, soc_max=1.0)

    os_leg_fixed = solve_lp_pyomo(
        price_ts=price_ts,
        prod1=prod_wind_leg, prod2=prod_pv_leg,
        stor1=stor_leg_fixed, stor2=stor_leg_null,
        discount_rate=0.03, n_year=20,
        p_max=p_grid, n=n,
        options=dict(name_solver='gurobi'))

    print(f"  [3] legacy, fixed prod + fixed storage")

    # Diagnostics: a timestep is "both-slack" when neither branch of the
    # relaxed storage dynamics is tight
    def _diag(os_, label):
        e = np.asarray(os_.storage_e[0].data, dtype=float)
        p = np.asarray(os_.storage_p[0].data, dtype=float)
        de = e[1:] - e[:-1]
        p_trim = p[:len(de)]

        slack_c = -(de + dt * eff * p_trim)
        slack_d = -(de + dt / eff * p_trim)

        tol = 1e-6
        both_slack = (slack_c > tol) & (slack_d > tol)
        n_off = int(both_slack.sum())

        de_phys = (-dt * eff * np.where(p_trim < 0, p_trim, 0.0)
                   - dt / eff * np.where(p_trim >= 0, p_trim, 0.0))
        fict = float(np.where(both_slack,
                              np.abs(de - de_phys), 0.0).sum())

        charge = float(-np.minimum(p, 0).sum() * dt)
        discharge = float(np.maximum(p, 0).sum() * dt)
        rte = (discharge / charge) if charge > 0 else float('nan')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            loss_ok = os_.check_losses(1e-4)

        print(f"\n      {label}")
        print(f"        both-slack       : {n_off} "
              f"({100 * n_off / len(both_slack):.2f}%)")
        print(f"        fictitious charge: {fict:.1f} MWh")
        print(f"        realized RTE     : {rte:.4f}")
        print(f"        check_losses     : {loss_ok}")

        return dict(n_off=n_off, fict=fict, rte=rte, loss_ok=loss_ok)

    d1 = _diag(os_sz,        "[1] sizing kernel (free prod, free stor)")
    d2 = _diag(os_leg_free,  "[2] legacy kernel (fixed prod, free stor)")
    d3 = _diag(os_leg_fixed, "[3] legacy kernel (fixed prod, fixed stor)")

    # Controls: the legacy kernel must be physically valid
    assert d2["loss_ok"] and d3["loss_ok"], (
        f"legacy kernel produced a physically invalid schedule "
        f"(fict = {d2['fict']:.1f} / {d3['fict']:.1f} MWh)"
    )

    # The sizing kernel must also be physically valid
    assert d1["loss_ok"], (
        f"sizing kernel produced a physically invalid schedule: "
        f"{d1['n_off']} both-slack hours, "
        f"{d1['fict']:.1f} MWh of fictitious charge, "
        f"realized RTE {d1['rte']:.4f} vs. nominal {eff * eff:.4f}"
    )

def test_sizing_degeneracy_tracks_headroom_binding():
    """Is the sizing kernel's storage degeneracy driven by the headroom
    constraint?

    Free production lets the LP shift the plant's operating point
    relative to p_grid_max, which changes where the headroom constraint
    binds, which changes the duals on the storage dynamics. This test
    sweeps p_grid_max and checks whether the fictitious charge tracks
    the fraction of hours at which headroom is binding.
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

    eff = 0.95

    def _fictitious(os_):
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

    def _headroom_fraction(os_, p_grid):
        A = np.asarray(os_.power_out.data) + np.asarray(
            os_.p_curtail.data)
        return float(np.mean(np.isclose(A, p_grid, atol=1e-3)))

    p_grid_values = [40.0, 60.0, 80.0, 100.0, 150.0]
    results = []

    for p_grid in p_grid_values:
        prod_wind = Production(prof_unit_ts, p_cost=1000000.0,
                               opex_fix=0.0, p_max=None,
                               p_max_is_decision=True)
        prod_pv = Production(pv_unit_ts, p_cost=1.0,
                             opex_fix=0.0, p_max=None,
                             p_max_is_decision=True)
        stor = Storage(e_cap=None, p_cap=None,
                       eff_in=eff, eff_out=eff,
                       p_cost=150_000.0, e_cost=150_000.0,
                       soc_min=0.0, soc_max=1.0, duration=2)
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
                options=dict(name_solver='gurobi'))

        n_off, fict = _fictitious(os_)
        headroom_frac = _headroom_fraction(os_, p_grid)
        x1 = os_.production_list[0].p_max
        results.append((p_grid, x1, n_off, fict, headroom_frac))

    print(f"\n  {'p_grid':>8} {'x1*':>8} {'both-slack':>12} "
          f"{'fict':>10} {'headroom%':>10}")
    for p_grid, x1, n_off, fict, hr in results:
        print(f"  {p_grid:>8.1f} {x1:>8.3f} {n_off:>12d} "
              f"{fict:>10.1f} {100 * hr:>9.2f}%")

    # Degeneracy must be present at the default p_grid = 80
    default = [r for r in results if r[0] == 80.0][0]
    assert default[3] > 1.0, (
        f"no degeneracy at p_grid = 80 (fict = {default[3]:.1f} MWh); "
        f"this test is only meaningful when the degeneracy is present"
    )

    # If headroom is the driver, the fictitious charge should be
    # non-monotonic in p_grid: it should peak somewhere in the range
    # where the LP is actively trading off production size against the
    # grid limit. If it's flat, headroom is not the driver.
    f_values = [r[3] for r in results]
    spread = max(f_values) - min(f_values)
    assert spread > 50.0, (
        f"fictitious charge is flat across p_grid (spread = "
        f"{spread:.1f} MWh); headroom binding is not the driver."
    )