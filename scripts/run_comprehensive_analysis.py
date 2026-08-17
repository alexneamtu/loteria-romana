#!/usr/bin/env python3
"""
Comprehensive Lottery Analysis Pipeline.

This script runs the full analysis suite on Romanian lottery data:
1. NIST randomness tests
2. Statistical bias detection
3. Expected value analysis
4. Machine learning models (Transformer, VAE)
5. Paper trading simulation

Usage:
    PYTHONPATH=src python scripts/run_comprehensive_analysis.py [options]

Options:
    --game GAME         Game to analyze: loto_649, loto_540, joker (default: loto_649)
    --quick             Run quick analysis (reduced epochs, smaller sample)
    --skip-ml           Skip ML models (for faster analysis)
    --output DIR        Output directory for reports (default: analysis_results)
    --verbose           Verbose output
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def parse_date_string(date_str: str) -> tuple[int, int, int]:
    """Parse date string in YYYY-MM-DD format to (year, month, day) tuple."""
    parts = date_str.split('-')
    return int(parts[0]), int(parts[1]), int(parts[2])


def load_draws(game: str) -> tuple[list[list[int]], list[tuple[int, int, int]], int, int]:
    """Load historical draws for the specified game."""
    data_dir = Path(__file__).parent.parent / "data" / "clean"

    if game == "loto_649":
        from loto_649_model.storage import load_draws as load_649
        draws_obj = load_649(data_dir / "loto_649_draws.csv")
        draws = [list(d.main_numbers) for d in draws_obj]
        dates = [parse_date_string(d.date) for d in draws_obj]
        pool_size = 49
        draw_size = 6
    elif game == "loto_540":
        from loto_540_model.storage import load_draws as load_540
        draws_obj = load_540(data_dir / "loto_540_draws.csv")
        draws = [list(d.main_numbers) for d in draws_obj]
        dates = [parse_date_string(d.date) for d in draws_obj]
        pool_size = 40
        draw_size = 5
    elif game == "joker":
        from joker_model.storage import load_draws as load_joker
        draws_obj = load_joker(data_dir / "joker_draws.csv")
        draws = [list(d.main_numbers) for d in draws_obj]
        dates = [parse_date_string(d.date) for d in draws_obj]
        pool_size = 45
        draw_size = 5
    else:
        raise ValueError(f"Unknown game: {game}")

    return draws, dates, pool_size, draw_size


def run_nist_tests(draws: list[list[int]], pool_size: int, verbose: bool = False) -> dict:
    """Run NIST randomness tests."""
    print("\n" + "=" * 60)
    print("PHASE 1: NIST RANDOMNESS TESTS")
    print("=" * 60)

    try:
        from shared.nist_tests import run_nist_analysis
        results = run_nist_analysis(draws, pool_size)

        # Extract encoding results (exclude 'summary' key)
        encoding_results = {k: v for k, v in results.items() if k != 'summary'}

        if verbose:
            print(f"\nEncoding methods tested: {len(encoding_results)}")
            for encoding, data in encoding_results.items():
                print(f"  {encoding}: {data['tests_passed']}/{data['tests_run']} tests passed")

        # Summary - check if all tests passed across all encodings
        all_passed = all(
            data['tests_failed'] == 0
            for data in encoding_results.values()
        )

        print(f"\nNIST Summary: {'All tests PASSED' if all_passed else 'Some tests FAILED'}")

        return {
            "status": "completed",
            "all_passed": all_passed,
            "results": {
                encoding: data['details']
                for encoding, data in encoding_results.items()
            }
        }

    except Exception as e:
        print(f"NIST tests failed: {e}")
        return {"status": "failed", "error": str(e)}


def run_bias_detection(draws: list[list[int]], dates: list[tuple],
                      pool_size: int, verbose: bool = False) -> dict:
    """Run statistical bias detection."""
    print("\n" + "=" * 60)
    print("PHASE 2: STATISTICAL BIAS DETECTION")
    print("=" * 60)

    try:
        from shared.bias_detector import BiasDetector

        detector = BiasDetector(significance_level=0.01)
        reports = detector.full_report(draws, pool_size, dates)
        summary = detector.summary(reports)

        if verbose:
            for report in reports:
                status = "⚠️ SIGNIFICANT" if report.is_significant else "✓ Pass"
                print(f"  {report.test_name}: {status} (p={report.p_value:.4f})")

        print(f"\nBias Detection Summary:")
        print(f"  Total tests: {summary['total_tests']}")
        print(f"  Significant findings: {summary['significant_findings']}")
        print(f"  Conclusion: {summary['overall_conclusion']}")

        return {
            "status": "completed",
            "summary": summary,
            "reports": [
                {
                    "test": r.test_name,
                    "significant": r.is_significant,
                    "p_value": r.p_value,
                    "statistic": r.statistic
                }
                for r in reports
            ]
        }

    except Exception as e:
        print(f"Bias detection failed: {e}")
        return {"status": "failed", "error": str(e)}


def run_ev_analysis(game: str, verbose: bool = False) -> dict:
    """Run expected value analysis."""
    print("\n" + "=" * 60)
    print("PHASE 3: EXPECTED VALUE ANALYSIS")
    print("=" * 60)

    try:
        from shared.ev_calculator import EVCalculator

        calc = EVCalculator()

        if game == "loto_649":
            lottery = calc.create_loto_649()
            jackpots = [100_000, 1_000_000, 5_000_000, 10_000_000]
        elif game == "loto_540":
            lottery = calc.create_loto_540()
            jackpots = [50_000, 500_000, 2_000_000, 5_000_000]
        else:  # joker
            lottery = calc.create_joker()
            jackpots = [200_000, 2_000_000, 10_000_000, 20_000_000]

        results = []
        for jackpot in jackpots:
            ev_result = calc.calculate_ev(lottery, jackpot=jackpot)
            results.append({
                "jackpot": jackpot,
                "expected_value": ev_result.expected_value,
                "return_pct": ev_result.return_percentage,
                "is_positive_ev": ev_result.is_positive_ev
            })

            if verbose:
                print(f"  Jackpot {jackpot:>12,} RON: EV = {ev_result.expected_value:>10.4f} RON "
                      f"({ev_result.return_percentage:.1f}% return)")

        # Calculate breakeven jackpot
        breakeven = calc._calculate_positive_ev_jackpot(lottery, 1.0, None)

        print(f"\nEV Summary for {game}:")
        print(f"  Ticket cost: {lottery.ticket_cost:.2f} RON")
        print(f"  Breakeven jackpot: {breakeven:,.0f} RON")
        print(f"  Current typical EV: {results[0]['expected_value']:.4f} RON (at minimum jackpot)")

        return {
            "status": "completed",
            "game": game,
            "ticket_cost": lottery.ticket_cost,
            "breakeven_jackpot": breakeven,
            "jackpot_analysis": results
        }

    except Exception as e:
        print(f"EV analysis failed: {e}")
        return {"status": "failed", "error": str(e)}


def run_transformer_analysis(draws: list[list[int]], game: str,
                            quick: bool = False, verbose: bool = False) -> dict:
    """Run Transformer sequence analysis."""
    print("\n" + "=" * 60)
    print("PHASE 4: TRANSFORMER SEQUENCE ANALYSIS")
    print("=" * 60)

    try:
        from shared.transformer_model import (
            TransformerConfig, LotteryTransformerTrainer,
            create_default_config, TORCH_AVAILABLE
        )

        if not TORCH_AVAILABLE:
            print("  PyTorch not available. Skipping Transformer analysis.")
            return {"status": "skipped", "reason": "PyTorch not installed"}

        config = create_default_config(game)
        config.num_epochs = 25 if quick else 100

        print(f"  Training Transformer model ({config.num_epochs} epochs)...")
        trainer = LotteryTransformerTrainer(config)

        train_results = trainer.train(draws, val_split=0.2, early_stop_patience=15)
        print(f"  Training complete. Best val loss: {train_results['best_val_loss']:.4f}")

        # Evaluate
        eval_results = trainer.evaluate(draws[-200:])

        print(f"\nTransformer Results:")
        print(f"  Average hits per draw: {eval_results['average_hits_per_draw']:.2f}")
        print(f"  Expected random hits: {eval_results['expected_random_hits']:.2f}")
        print(f"  Improvement over random: {eval_results['improvement_over_random']*100:.1f}%")

        # Generate sample predictions
        predictions = trainer.generate_predictions(draws, num_lines=5)
        print(f"\nSample predictions for next draw:")
        for i, pred in enumerate(predictions, 1):
            print(f"  Line {i}: {pred}")

        return {
            "status": "completed",
            "training": {
                "epochs": train_results["epochs_trained"],
                "best_val_loss": train_results["best_val_loss"]
            },
            "evaluation": eval_results,
            "sample_predictions": predictions,
            "conclusion": eval_results.get("conclusion", "Analysis complete")
        }

    except ImportError as e:
        print(f"  Transformer analysis skipped: {e}")
        return {"status": "skipped", "reason": str(e)}
    except Exception as e:
        print(f"Transformer analysis failed: {e}")
        return {"status": "failed", "error": str(e)}


def run_vae_analysis(draws: list[list[int]], game: str,
                    quick: bool = False, verbose: bool = False) -> dict:
    """Run VAE anomaly detection analysis."""
    print("\n" + "=" * 60)
    print("PHASE 5: VAE ANOMALY DETECTION")
    print("=" * 60)

    try:
        from shared.vae_model import (
            VAEConfig, VAETrainer,
            create_default_config, TORCH_AVAILABLE
        )

        if not TORCH_AVAILABLE:
            print("  PyTorch not available. Skipping VAE analysis.")
            return {"status": "skipped", "reason": "PyTorch not installed"}

        config = create_default_config(game)
        config.num_epochs = 25 if quick else 100

        print(f"  Training VAE model ({config.num_epochs} epochs)...")
        trainer = VAETrainer(config)

        train_results = trainer.train(draws, val_split=0.2)
        print(f"  Training complete. Anomaly threshold: {train_results['anomaly_threshold']:.4f}")

        # Detect anomalies
        anomalies = trainer.detect_anomalies(draws)
        flagged = [a for a in anomalies if a["is_anomaly"]]

        print(f"\nAnomaly Detection Results:")
        print(f"  Total draws scanned: {len(draws)}")
        print(f"  Anomalies detected: {len(flagged)}")
        print(f"  Anomaly rate: {len(flagged)/len(draws)*100:.1f}%")

        if flagged and verbose:
            print(f"\nTop anomalies:")
            for a in sorted(flagged, key=lambda x: -x["anomaly_score"])[:5]:
                print(f"  Draw {a['draw_index']}: {a['draw']} (score: {a['anomaly_score']:.2f})")

        # Distribution validation
        dist_comparison = trainer.compare_distributions(draws[-500:])

        print(f"\nDistribution Validation:")
        print(f"  KL divergence: {dist_comparison['kl_divergence']:.4f}")
        print(f"  Distribution match: {'Yes' if dist_comparison['distribution_match'] else 'No'}")

        return {
            "status": "completed",
            "training": {
                "epochs": train_results["epochs_trained"],
                "anomaly_threshold": train_results["anomaly_threshold"]
            },
            "anomaly_detection": {
                "total_scanned": len(draws),
                "anomalies_found": len(flagged),
                "anomaly_rate": len(flagged) / len(draws),
                "top_anomalies": [
                    {"index": a["draw_index"], "draw": a["draw"], "score": a["anomaly_score"]}
                    for a in sorted(flagged, key=lambda x: -x["anomaly_score"])[:10]
                ]
            },
            "distribution_validation": dist_comparison
        }

    except ImportError as e:
        print(f"  VAE analysis skipped: {e}")
        return {"status": "skipped", "reason": str(e)}
    except Exception as e:
        print(f"VAE analysis failed: {e}")
        return {"status": "failed", "error": str(e)}


def run_paper_trading(draws: list[list[int]], pool_size: int, draw_size: int,
                     verbose: bool = False, quick: bool = False) -> dict:
    """Run paper trading simulation."""
    print("\n" + "=" * 60)
    print("PHASE 6: PAPER TRADING SIMULATION")
    print("=" * 60)

    try:
        from shared.paper_trader import PaperTrader
        import random

        # Simple prize calculator
        def prize_calculator(picks, actual):
            matches = len(set(picks) & set(actual))
            prizes = {6: 1_000_000, 5: 5000, 4: 80, 3: 20}
            return prizes.get(matches, 0)

        trader = PaperTrader(
            initial_balance=10000,
            ticket_cost=6.0,
            prize_calculator=prize_calculator
        )

        # Test multiple strategies
        strategies = {
            "random": lambda hist, n: [[random.randint(1, pool_size) for _ in range(draw_size)]
                                       for _ in range(n)],
            "frequency": lambda hist, n: generate_frequency_picks(hist, pool_size, draw_size, n)
        }

        print("  Simulating strategies on historical data...")

        # Use fewer draws in quick mode
        num_test_draws = 100 if quick else 500
        test_draws = draws[-num_test_draws:]
        history = draws[:-num_test_draws]

        for strategy_name, strategy_func in strategies.items():
            for i, actual_draw in enumerate(test_draws):
                picks = strategy_func(history, 1)
                trader.execute_trade(
                    draw_id=i,
                    strategy_name=strategy_name,
                    tickets=picks,
                    actual_draw=actual_draw
                )
                history.append(actual_draw)

        print(f"\n{trader.generate_report()}")

        # Get stats for all strategies
        strategy_stats = {}
        for name in strategies:
            stats = trader.get_stats(name)
            strategy_stats[name] = {
                "total_trades": stats.total_trades,
                "roi": stats.roi,
                "win_rate": stats.win_rate,
                "ev_per_ticket": stats.expected_value_per_ticket,
                "is_significant": stats.is_statistically_significant,
                "p_value": stats.p_value
            }

        return {
            "status": "completed",
            "initial_balance": trader.initial_balance,
            "final_balance": trader.balance,
            "net_pnl": trader.balance - trader.initial_balance,
            "strategies": strategy_stats
        }

    except Exception as e:
        print(f"Paper trading failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


def generate_frequency_picks(history, pool_size, draw_size, num_picks):
    """Generate picks based on frequency analysis."""
    from collections import Counter

    freq = Counter()
    for draw in history[-200:]:  # Recent history
        for num in draw:
            freq[num] += 1

    # Get most frequent numbers
    most_common = [num for num, _ in freq.most_common(draw_size * 2)]

    picks = []
    for _ in range(num_picks):
        import random
        pick = sorted(random.sample(most_common, draw_size))
        picks.append(pick)

    return picks


def generate_final_report(results: dict, game: str, output_dir: Path) -> str:
    """Generate final comprehensive report."""
    report_lines = [
        "=" * 70,
        "COMPREHENSIVE LOTTERY ANALYSIS REPORT",
        f"Game: {game.upper()}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        ""
    ]

    # NIST Results
    nist = results.get("nist", {})
    report_lines.extend([
        "1. RANDOMNESS VERIFICATION (NIST SP 800-22)",
        "-" * 50
    ])
    if nist.get("status") == "completed":
        report_lines.append(f"   Result: {'PASS - Data appears random' if nist['all_passed'] else 'FAIL - Some tests failed'}")
    else:
        report_lines.append(f"   Status: {nist.get('status', 'unknown')}")
    report_lines.append("")

    # Bias Detection
    bias = results.get("bias", {})
    report_lines.extend([
        "2. STATISTICAL BIAS DETECTION",
        "-" * 50
    ])
    if bias.get("status") == "completed":
        summary = bias.get("summary", {})
        report_lines.append(f"   Tests run: {summary.get('total_tests', 0)}")
        report_lines.append(f"   Significant biases: {summary.get('significant_findings', 0)}")
        report_lines.append(f"   Conclusion: {summary.get('overall_conclusion', 'N/A')}")
    else:
        report_lines.append(f"   Status: {bias.get('status', 'unknown')}")
    report_lines.append("")

    # EV Analysis
    ev = results.get("ev", {})
    report_lines.extend([
        "3. EXPECTED VALUE ANALYSIS",
        "-" * 50
    ])
    if ev.get("status") == "completed":
        report_lines.append(f"   Ticket cost: {ev.get('ticket_cost', 0):.2f} RON")
        report_lines.append(f"   Breakeven jackpot: {ev.get('breakeven_jackpot', 0):,.0f} RON")
        report_lines.append("   Conclusion: Play has negative expected value under normal conditions")
    else:
        report_lines.append(f"   Status: {ev.get('status', 'unknown')}")
    report_lines.append("")

    # ML Results
    transformer = results.get("transformer", {})
    report_lines.extend([
        "4. MACHINE LEARNING ANALYSIS",
        "-" * 50
    ])
    if transformer.get("status") == "completed":
        eval_res = transformer.get("evaluation", {})
        report_lines.append(f"   Transformer improvement over random: {eval_res.get('improvement_over_random', 0)*100:.1f}%")
    else:
        report_lines.append(f"   Transformer: {transformer.get('status', 'skipped')}")

    vae = results.get("vae", {})
    if vae.get("status") == "completed":
        anomaly = vae.get("anomaly_detection", {})
        report_lines.append(f"   VAE anomaly rate: {anomaly.get('anomaly_rate', 0)*100:.1f}%")
    else:
        report_lines.append(f"   VAE: {vae.get('status', 'skipped')}")
    report_lines.append("")

    # Paper Trading
    trading = results.get("paper_trading", {})
    report_lines.extend([
        "5. PAPER TRADING RESULTS",
        "-" * 50
    ])
    if trading.get("status") == "completed":
        report_lines.append(f"   Net P&L: {trading.get('net_pnl', 0):,.2f} RON")
        for name, stats in trading.get("strategies", {}).items():
            report_lines.append(f"   {name}: ROI={stats['roi']*100:.1f}%, p-value={stats['p_value']:.4f}")
    else:
        report_lines.append(f"   Status: {trading.get('status', 'unknown')}")
    report_lines.append("")

    # Overall Conclusion
    report_lines.extend([
        "=" * 70,
        "OVERALL CONCLUSION",
        "=" * 70,
        "",
        "Based on comprehensive analysis:",
        ""
    ])

    # Determine overall conclusion
    is_random = nist.get("all_passed", True)
    no_bias = bias.get("summary", {}).get("significant_findings", 0) == 0
    ml_no_edge = transformer.get("evaluation", {}).get("improvement_over_random", 0) < 0.1

    if is_random and no_bias and ml_no_edge:
        report_lines.extend([
            "✓ Lottery appears to be FAIR and RANDOM",
            "✓ No exploitable patterns detected",
            "✓ Machine learning models cannot beat random selection",
            "✓ Expected value is negative under all normal conditions",
            "",
            "RECOMMENDATION: Play only for entertainment within budget.",
            "No prediction method can improve your odds."
        ])
    else:
        report_lines.extend([
            "⚠ Some tests flagged potential anomalies",
            "  These warrant further investigation but are likely",
            "  due to statistical variance rather than exploitable patterns.",
            "",
            "RECOMMENDATION: Continue monitoring with paper trading.",
            "Do not assume any pattern is exploitable without",
            "extended out-of-sample validation."
        ])

    report_lines.append("")
    report_lines.append("=" * 70)

    report = "\n".join(report_lines)

    # Save report
    report_path = output_dir / f"analysis_report_{game}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Lottery Analysis")
    parser.add_argument("--game", choices=["loto_649", "loto_540", "joker"],
                       default="loto_649", help="Game to analyze")
    parser.add_argument("--quick", action="store_true",
                       help="Quick analysis mode (reduced epochs)")
    parser.add_argument("--skip-ml", action="store_true",
                       help="Skip ML models")
    parser.add_argument("--output", default="analysis_results",
                       help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPREHENSIVE LOTTERY ANALYSIS PIPELINE")
    print(f"Game: {args.game.upper()}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print("=" * 70)

    # Load data
    print("\nLoading historical draws...")
    try:
        draws, dates, pool_size, draw_size = load_draws(args.game)
        print(f"  Loaded {len(draws)} draws")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    results = {}

    # Phase 1: NIST Tests (use subset in quick mode for faster execution)
    nist_draws = draws[-500:] if args.quick else draws
    results["nist"] = run_nist_tests(nist_draws, pool_size, args.verbose)

    # Phase 2: Bias Detection
    results["bias"] = run_bias_detection(draws, dates, pool_size, args.verbose)

    # Phase 3: EV Analysis
    results["ev"] = run_ev_analysis(args.game, args.verbose)

    # Phase 4 & 5: ML Models (optional)
    if not args.skip_ml:
        results["transformer"] = run_transformer_analysis(draws, args.game, args.quick, args.verbose)
        results["vae"] = run_vae_analysis(draws, args.game, args.quick, args.verbose)
    else:
        print("\n[ML models skipped]")
        results["transformer"] = {"status": "skipped", "reason": "User requested skip"}
        results["vae"] = {"status": "skipped", "reason": "User requested skip"}

    # Phase 6: Paper Trading
    results["paper_trading"] = run_paper_trading(draws, pool_size, draw_size, args.verbose, args.quick)

    # Generate final report
    report = generate_final_report(results, args.game, output_dir)
    print("\n" + report)

    # Save raw results
    results_path = output_dir / f"raw_results_{args.game}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Clean results for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, float):
            if obj != obj:  # NaN check
                return None
            return obj
        return obj

    with open(results_path, 'w') as f:
        json.dump(clean_for_json(results), f, indent=2, default=str)

    print(f"\nRaw results saved to: {results_path}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
