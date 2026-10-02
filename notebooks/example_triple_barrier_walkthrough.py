#!/usr/bin/env python3
"""Atlas triple barrier: data → illustration → walk-forward CV → trades → worker.

Run from the repository root with notebooks/.venv/bin/python. Output is kept in
notebooks/triple_barrier_example_output/; deploy its predict.pkl with notebooks/deploy_worker.py.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator

TOPICS = {87: 'hl_xyzgold_1min', 88: 'hl_xyzsilver_1min', 89: 'hl_xyzcl_1min'}
CLASSES = ['down', 'neutral', 'up']
TARGETS = ['target_' + c for c in CLASSES]
PREDS = ['pred_' + c for c in CLASSES]
INTERVAL = '1h'
TARGET_BARS = 24
INPUT_BARS = 100


def api_key():
    """Runtime credential lookup; secrets are never serialized into the model."""
    value = os.environ.get('ALLORA_API_KEY', '').strip()
    if value:
        return value
    for path in (Path.cwd() / '.allora_api_key', Path(__file__).resolve().parent / '.allora_api_key',
                 Path(__file__).resolve().parents[1] / '.allora_api_key'):
        if path.exists():
            value = path.read_text().strip()
            if value:
                return value
    raise RuntimeError('Set ALLORA_API_KEY to your authorized Atlas/Allora key')


def make_workflow(topic, cache_dir):
    key = api_key()
    os.environ['ALLORA_API_KEY'] = key
    return AlloraMLWorkflow(tickers=[TOPICS[topic]], number_of_input_bars=INPUT_BARS,
                            target_bars=TARGET_BARS, interval=INTERVAL,
                            target_type='triple_barrier', data_source='allora',
                            api_key=key, base_dir=str(cache_dir))


def model_probabilities(model, features):
    probabilities = np.zeros((len(features), 3))
    probabilities[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(features)
    return PerformanceEvaluator.validate_probabilities(probabilities)


def make_predict(model, features, ticker, input_bars):
    """Self-contained callable compatible with the existing nonce artifact flow."""
    # Capture training configuration, never a data manager/API credential.
    interval, target_bars = INTERVAL, TARGET_BARS
    def predict(nonce: int = None):
        import os
        import numpy as np
        from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator
        key = os.environ.get('ALLORA_API_KEY', '').strip()
        if not key:
            raise RuntimeError('ALLORA_API_KEY is required for live Atlas features')
        workflow = AlloraMLWorkflow(
            tickers=[ticker], number_of_input_bars=input_bars, target_bars=target_bars, interval=interval,
            target_type='triple_barrier', data_source='allora', api_key=key,
        )
        live = workflow.get_live_features(ticker)[features]
        probabilities = np.zeros((len(live), 3))
        probabilities[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(live)
        PerformanceEvaluator.validate_probabilities(probabilities)
        return dict(zip(('down', 'neutral', 'up'), map(float, probabilities[-1])))
    return predict


def configure_plot_style():
    """Readable at blog width; plotting remains outside the inference artifact."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 15, 'axes.titlesize': 20, 'axes.titleweight': 'semibold',
        'axes.titlepad': 18, 'axes.labelsize': 15, 'axes.labelpad': 10,
        'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13,
        'figure.titlesize': 22, 'figure.titleweight': 'semibold',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#bcc4cc', 'text.color': '#17212b',
        'axes.labelcolor': '#17212b', 'xtick.color': '#364452', 'ytick.color': '#364452',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'savefig.dpi': 200, 'lines.linewidth': 2,
    })


def date_axis(ax, intraday=False):
    """Use a bounded number of short date labels, never crowded ISO strings."""
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M' if intraday else '%b %d',
                                                      tz=timezone.utc))
    ax.tick_params(axis='x', pad=9)
    ax.grid(alpha=.18)
    ax.set_axisbelow(True)


def save_plot(fig, output, name):
    fig.savefig(output / name, dpi=200, bbox_inches='tight', pad_inches=.2)


def candles(ax, bars, volume_ax=None, current=None):
    """Draw native hourly candles, preserving their OPEN timestamps."""
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    for row in bars.itertuples():
        x = mdates.date2num(row.open_time)
        color = '#87909a' if current is not None and row.open_time < current else (
            '#16876c' if row.close >= row.open else '#c84953')
        ax.vlines(x + 1/48, row.low, row.high, color=color, lw=.8)
        body = max(abs(row.close - row.open), row.close * .000005)
        ax.add_patch(Rectangle((x + .1/24, min(row.open, row.close)), .8/24, body,
                               facecolor=color, edgecolor=color, alpha=.85))
        if volume_ax is not None:
            volume_ax.bar(x + 1/48, row.volume, width=.8/24, color=color, alpha=.65)
    ax.autoscale_view()
    ax.xaxis_date()
    date_axis(ax, intraday=True)
    ax.set_ylabel('Price (USD)')


def draw_barriers(ax, row, entry_direction=None):
    import matplotlib.dates as mdates
    t = row.tb_prediction_time
    end = t + pd.Timedelta(hours=TARGET_BARS)
    for price, color, label in [(row.tb_upper, '#16876c', 'Upper barrier'),
                                (row.tb_lower, '#c84953', 'Lower barrier'),
                                (row.tb_base, '#207ca8', 'Entry / base price')]:
        ax.hlines(price, mdates.date2num(t), mdates.date2num(end),
                  color=color, linestyle='--', linewidth=1.8, label=label)
    ax.axvline(t, color='#207ca8', lw=1.5, label='Now: candle close')
    ax.axvline(end, color='#ad7600', linestyle=':', lw=2, label='24h time boundary')
    ax.axvspan(t, end, color='#207ca8', alpha=.035)
    ax.scatter(row.tb_exit_time, row.tb_exit_price, color='#17212b',
               edgecolors='white', linewidths=1.2, zorder=6, s=70)
    # Reserve a clear band above the candles for callouts. Opaque boxes and
    # axes-relative anchors prevent labels from disappearing into candle bodies.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + .35 * (hi - lo))
    reason = {'up': 'Upper barrier', 'down': 'Lower barrier', 'expiry': 'Time expiry'}[row.tb_exit_reason]
    ax.annotate(f'{"Exit" if entry_direction is not None else "First touch"} · {reason}\n'
                f'{row.tb_exit_time:%b %d, %H:%M} UTC · ${row.tb_exit_price:,.2f}',
                (mdates.date2num(row.tb_exit_time), row.tb_exit_price),
                xytext=(.98, .97), textcoords='axes fraction', ha='right', va='top',
                fontsize=13, linespacing=1.5, zorder=10,
                bbox=dict(boxstyle='round,pad=.55', facecolor='white', edgecolor='#697785', alpha=1),
                arrowprops=dict(arrowstyle='->', color='#364452', lw=1.5, shrinkA=7, shrinkB=7))
    if entry_direction is not None:
        ax.scatter(t, row.tb_base, marker='^' if entry_direction == 1 else 'v',
                   color='#295ab0', edgecolors='white', linewidths=1, s=110, zorder=7)
        ax.annotate(f'Entry · {"Long" if entry_direction == 1 else "Short"}\n'
                    f'{t:%b %d, %H:%M} UTC · ${row.tb_base:,.2f}',
                    (mdates.date2num(t), row.tb_base),
                    xytext=(.02, .97), textcoords='axes fraction', ha='left', va='top',
                    fontsize=13, linespacing=1.5, zorder=10,
                    bbox=dict(boxstyle='round,pad=.55', facecolor='white', edgecolor='#295ab0', alpha=1),
                    arrowprops=dict(arrowstyle='->', color='#295ab0', lw=1.5, shrinkA=7, shrinkB=7))


def plot_intro(data, resolved, output, show=False):
    import matplotlib.pyplot as plt
    directional = resolved[resolved.target_neutral == 0]
    row = directional.iloc[len(directional) // 2] if len(directional) else resolved.iloc[len(resolved) // 2]
    t = row.tb_prediction_time
    view = data[(data.open_time >= t - pd.Timedelta(hours=INPUT_BARS)) &
                (data.open_time < t + pd.Timedelta(hours=TARGET_BARS))]
    fig, (ax, vol) = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, layout='constrained',
                                 gridspec_kw={'height_ratios': [4, 1], 'hspace': .08})
    candles(ax, view, vol, current=t)
    draw_barriers(ax, row)
    target = CLASSES[np.argmax(row[TARGETS].to_numpy(dtype=float))]
    asset = {'hl_xyzgold_1min': 'Gold', 'hl_xyzsilver_1min': 'Silver', 'hl_xyzcl_1min': 'WTI oil'}.get(row.ticker, row.ticker)
    fig.suptitle(f'{asset}: the triple-barrier learning problem\n', x=.08, ha='left')
    ax.set_title(f'{INPUT_BARS}h of observed history → 24h prediction horizon · target: {target}',
                 fontsize=15, fontweight='normal', loc='left', pad=16)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.12), ncol=3,
              frameon=False, fontsize=12)
    vol.set_ylabel('Volume')
    vol.set_xlabel(f'Hourly candles · UTC ({t.year}) · first touch from minute data')
    save_plot(fig, output, 'triple_barrier_example.png')
    if show:
        plt.show()
    plt.close(fig)


def trade_ledger(holdout, size=1., cost_bps=0.):
    """Book exact minute-tested barriers or last in-window close at expiration.

    The target builder has already replayed each minute path and retained its
    first-touch/expiry price. Reuse that evidence, not an hourly approximation.
    Trades overlap; fixed units are not a capital-normalized portfolio return.
    """
    columns = ['entry_time', 'exit_time', 'direction', 'entry_price', 'exit_price',
               'upper', 'lower', 'exit_reason', 'size', 'gross_pnl', 'cost', 'net_pnl']
    rows = []
    signals = holdout[PREDS].to_numpy().argmax(1) - 1
    for (_, row), direction in zip(holdout.iterrows(), signals):
        if direction == 0:
            continue
        gross = direction * (row.tb_exit_price - row.tb_base) * size
        cost = (row.tb_base + row.tb_exit_price) * size * cost_bps / 10000
        rows.append(dict(entry_time=row.tb_prediction_time, exit_time=row.tb_exit_time,
                         direction=int(direction), entry_price=row.tb_base, exit_price=row.tb_exit_price,
                         upper=row.tb_upper, lower=row.tb_lower, exit_reason=row.tb_exit_reason,
                         size=size, gross_pnl=gross, cost=cost, net_pnl=gross-cost))
    return pd.DataFrame(rows, columns=columns).sort_values('exit_time', kind='stable')


def plot_results(data, holdout, trades, output, diagnostic_cost, trade_cost_bps):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    y = holdout[TARGETS].to_numpy().argmax(1)
    hard = holdout[PREDS].to_numpy().argmax(1)
    year = holdout.tb_prediction_time.iloc[0].year
    fig, ax = plt.subplots(figsize=(8, 7), layout='constrained')
    display = ConfusionMatrixDisplay(confusion_matrix(y, hard, labels=[0, 1, 2]), display_labels=CLASSES)
    display.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d',
                 text_kw={'fontsize': 23, 'fontweight': 'semibold'})
    ax.set_title('Final holdout: predicted vs. actual class', pad=22)
    ax.tick_params(labelsize=16)
    save_plot(fig, output, 'confusion_matrix.png'); plt.close(fig)
    payoff = (hard-1)*(y-1) - diagnostic_cost*(hard != 1)
    fig, ax = plt.subplots(figsize=(12, 5.5), layout='constrained')
    ax.plot(holdout.tb_prediction_time, payoff.cumsum(), color='#397ca8', lw=2.3)
    ax.set(title=f'Directional classification payoff\nCost: {diagnostic_cost:g} barrier units per trade',
           ylabel='Cumulative payoff (barrier units)', xlabel=f'Prediction date · UTC ({year})')
    date_axis(ax)
    save_plot(fig, output, 'directional_payoff.png'); plt.close(fig)
    selected = holdout.loc[hard != 1]
    fig, axes = plt.subplots(3, 1, figsize=(13, 15), layout='constrained')
    fig.set_constrained_layout_pads(hspace=.12)
    fig.suptitle('Example trades from the final holdout', x=.08, ha='left')
    if len(selected):
        positions = np.unique(np.linspace(0, len(selected)-1, min(3, len(selected)), dtype=int))
        for number, (ax, pos) in enumerate(zip(axes, positions), start=1):
            row = selected.iloc[pos]
            t = row.tb_prediction_time
            view = data[(data.open_time >= t-pd.Timedelta(hours=6)) &
                        (data.open_time < t+pd.Timedelta(hours=TARGET_BARS))]
            direction = int(np.argmax(row[PREDS].to_numpy(dtype=float))) - 1
            candles(ax, view, current=t)
            draw_barriers(ax, row, entry_direction=direction)
            ax.set_title(f'Trade {number} · {"Long" if direction == 1 else "Short"}', loc='left', fontsize=18)
            ax.set_xlabel(f'Hourly candles · UTC ({year})')
        for ax in axes[len(positions):]:
            ax.set_visible(False)
    else:
        axes[0].text(.5, .5, 'No directional predictions in this holdout', ha='center', transform=axes[0].transAxes)
    save_plot(fig, output, 'example_trades.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
    if len(trades):
        cumulative = trades.groupby('exit_time')[['gross_pnl', 'net_pnl']].sum().cumsum()
        ax.step(cumulative.index, cumulative.gross_pnl, where='post', label='Gross', color='#397ca8', lw=2.5)
        ax.step(cumulative.index, cumulative.net_pnl, where='post', label='Net', color='#cc7825', lw=2)
        ax.legend(loc='upper left', framealpha=1, facecolor='white')
    ax.set(title=f'Final holdout: cumulative trading PnL\n1 unit per trade · {trade_cost_bps:g} bps per side · overlapping positions',
           ylabel='Cumulative realized PnL (USD)', xlabel=f'Exit date · UTC ({year})')
    date_axis(ax)
    save_plot(fig, output, 'cumulative_trade_pnl.png'); plt.close(fig)


def plot_folds(samples, splits, output):
    """Show the actual training/validation dates, including excluded label gaps."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(13, 7), layout='constrained')
    periods = []
    width = pd.Timedelta(hours=1)
    for i, (train, valid) in enumerate(splits):
        train_start = samples.iloc[train[0]].tb_prediction_time
        train_end = samples.iloc[train[-1]].tb_prediction_time + width
        valid_start = samples.iloc[valid[0]].tb_prediction_time
        valid_end = samples.iloc[valid[-1]].tb_prediction_time + width
        final = i == len(splits)-1
        for start, end, color in [(train_start, train_end, '#397ca8'),
                                   (valid_start, valid_end, '#9254a1' if final else '#269b80')]:
            left, right = mdates.date2num(start), mdates.date2num(end)
            ax.barh(i, right-left, left=left, height=.55, color=color)
        if train_end < valid_start:
            left, right = mdates.date2num(train_end), mdates.date2num(valid_start)
            ax.barh(i, right-left, left=left, height=.55, facecolor='#eeeeee',
                    edgecolor='#888888', hatch='///')
        ax.text((mdates.date2num(train_start)+mdates.date2num(train_end))/2, i,
                f'{len(train):,} train', fontsize=14, color='white', weight='semibold', ha='center', va='center')
        ax.text((mdates.date2num(valid_start)+mdates.date2num(valid_end))/2, i,
                f'{len(valid):,} OOS', fontsize=14, color='white', weight='semibold', ha='center', va='center')
        periods.append(dict(fold=i+1, role='final_holdout' if final else 'model_selection',
                            train_start=train_start.isoformat(), train_end_exclusive=train_end.isoformat(),
                            validation_start=valid_start.isoformat(), validation_end_exclusive=valid_end.isoformat(),
                            train_rows=len(train), validation_rows=len(valid)))
    ax.set_yticks(range(len(splits)), [f'Fold {i+1}' + (' — final holdout' if i == len(splits)-1 else '')
                                     for i in range(len(splits))])
    ax.invert_yaxis()
    ax.xaxis_date()
    date_axis(ax)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel(f'Prediction date · UTC ({samples.tb_prediction_time.iloc[0].year})')
    ax.set_title('Walk-forward model selection\nEarlier folds: minimize log loss · final fold: OOS evaluation', pad=24)
    ax.legend(handles=[Patch(color='#397ca8', label='Training'),
                       Patch(color='#269b80', label='CV validation'),
                       Patch(color='#9254a1', label='Final OOS holdout'),
                       Patch(facecolor='#eeeeee', edgecolor='#888888', hatch='///', label='Unresolved-label gap')],
              loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2, frameon=False, fontsize=13)
    save_plot(fig, output, 'walk_forward_folds.png')
    plt.close(fig)
    return periods


def run(args):
    configure_plot_style()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir).resolve() if args.cache_dir else output / 'cache'
    workflow = make_workflow(args.topic, cache)
    if not args.skip_backfill:
        workflow.backfill(start=datetime.now(timezone.utc)-timedelta(days=args.days))
    data = workflow.get_full_feature_target_dataframe().reset_index()
    resolved = data.dropna(subset=TARGETS).copy()
    if resolved.empty:
        raise ValueError('No resolved targets; include 100 horizons plus one native bar of warmup and complete minute coverage')
    print(f'Loaded {len(data)} rows; {len(resolved)} resolved; class totals: {resolved[TARGETS].sum().to_dict()}')
    plot_intro(data, resolved, output, args.show)
    features = [c for c in data if c.startswith('feature_')]
    samples = resolved.dropna(subset=features).copy().reset_index(drop=True)
    samples = samples[np.isfinite(samples[features]).all(axis=1)].reset_index(drop=True)
    if len(samples) < args.folds + 1:
        raise ValueError('Not enough complete feature rows for walk-forward folds')
    truth = samples[TARGETS].to_numpy().argmax(1)
    splits = []
    for train, valid in TimeSeriesSplit(n_splits=args.folds).split(samples):
        # Conservatively require full-horizon coverage, even for earlier hits;
        # tb_exit_time is the trading exit, not confirmed network availability.
        # Exclude training labels unavailable at the validation cutoff, and use
        # these exact indices for both the timeline and model fitting.
        train = train[(samples.iloc[train].tb_resolution_time < samples.iloc[valid[0]].tb_prediction_time).to_numpy()]
        if not len(train):
            raise ValueError('Training window has no resolved labels before validation')
        splits.append((train, valid))
    fold_periods = plot_folds(samples, splits, output)
    configurations = [dict(n_estimators=n, num_leaves=leaves) for n, leaves in [(80, 7), (150, 15), (250, 15)]]
    cv_results = []
    for config in configurations:
        losses = []
        for train, valid in splits[:-1]:
            model = LGBMClassifier(**config, learning_rate=.03, max_depth=5, min_child_samples=30,
                                   random_state=42, n_jobs=2, verbosity=-1)
            model.fit(samples.iloc[train][features], truth[train])
            p = model_probabilities(model, samples.iloc[valid][features])
            losses.append(float(log_loss(truth[valid], p, labels=[0, 1, 2])))
        cv_results.append(dict(params=config, mean_log_loss=float(np.mean(losses)), fold_losses=losses))
    best = min(cv_results, key=lambda r: r['mean_log_loss'])
    print(f"Selected by earlier-fold mean log loss: {best['mean_log_loss']:.6f}; {best['params']}")
    train, valid = splits[-1]
    model = LGBMClassifier(**best['params'], learning_rate=.03, max_depth=5, min_child_samples=30,
                           random_state=42, n_jobs=2, verbosity=-1)
    model.fit(samples.iloc[train][features], truth[train])
    holdout = samples.iloc[valid].copy()
    holdout[PREDS] = model_probabilities(model, holdout[features])
    evaluator = PerformanceEvaluator(target_type='triple_barrier')
    baseline = evaluator.causal_class_baseline(holdout.tb_prediction_time, resolved.tb_resolution_time, resolved[TARGETS])
    holdout[['baseline_'+c for c in CLASSES]] = baseline
    available = data[(data.open_time >= holdout.open_time.min()) & (data.open_time <= holdout.open_time.max())]
    report = evaluator.evaluate(holdout[TARGETS], holdout[PREDS], baseline_probabilities=baseline,
                                n_expected_epochs=len(available), participation_kind='offline',
                                transaction_cost=args.diagnostic_cost)
    evaluator.print_report(report)
    holdout.to_csv(output / 'predictions.csv', index=False)
    trades = trade_ledger(holdout, cost_bps=args.trade_cost_bps)
    trades.to_csv(output / 'trades.csv', index=False)
    plot_results(data, holdout, trades, output, args.diagnostic_cost, args.trade_cost_bps)
    config = dict(topic=args.topic, dataset=TOPICS[args.topic], interval=INTERVAL, target_bars=TARGET_BARS,
                  input_bars=INPUT_BARS, atr_lookback_horizons=100, barrier_multiplier=.25,
                  cv=cv_results, best_params=best['params'], selection='lowest mean log loss on earlier folds only; final fold held out',
                  class_order=CLASSES, features=features, trade_size=1., trade_cost_bps=args.trade_cost_bps,
                  diagnostic_cost=args.diagnostic_cost, sdk='1.4.0rc4', folds=args.folds, fold_periods=fold_periods)
    for name, content in [('metrics.json', report), ('config.json', config)]:
        (output / name).write_text(json.dumps(content, indent=2, allow_nan=False))
    (output / 'report.txt').write_text('Final-fold OOS classification and trades\n' + json.dumps(report, indent=2) +
                                     f'\nTrades: {len(trades)}; net PnL: {trades.net_pnl.sum():.6f} USD\n'
                                     'Fixed one-unit overlapping trades; exact boundary fills, expiry at last in-window close.\n'
                                     'Participation is offline coverage, not observed network participation.\n')
    # Holdout figures above remain OOS; production refit does not overwrite them.
    model.fit(samples[features], truth)
    predict = make_predict(model, features, TOPICS[args.topic], INPUT_BARS)
    with (output / 'predict.pkl').open('wb') as handle:
        cloudpickle.dump(predict, handle)
    print(f'COMPLETE: {output}')
    print(f'TOPIC_ID={args.topic} PREDICT_PKL={output / "predict.pkl"} python notebooks/deploy_worker.py')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', type=int, choices=TOPICS, default=87)
    parser.add_argument('--days', type=int, default=220)
    parser.add_argument('--folds', type=int, default=4)
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent / 'triple_barrier_example_output'),
                        help='Artifacts directory; repeated runs replace outputs here (override to retain separate runs)')
    parser.add_argument('--cache-dir')
    parser.add_argument('--skip-backfill', action='store_true', help='Use existing normal Atlas cache')
    parser.add_argument('--show', action='store_true', help='Also display the first illustration interactively')
    parser.add_argument('--diagnostic-cost', type=float, default=0., help='Cost per directional signal in barrier units')
    parser.add_argument('--trade-cost-bps', type=float, default=0., help='Trading cost per side in basis points')
    args = parser.parse_args()
    if args.folds < 2 or min(args.diagnostic_cost, args.trade_cost_bps) < 0:
        parser.error('At least two folds and non-negative costs required')
    run(args)


if __name__ == '__main__':
    main()
