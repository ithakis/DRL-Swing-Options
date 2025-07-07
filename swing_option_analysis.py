#!/usr/bin/env python3
"""
Swing Option Value/Rights Analysis
Quantitative Finance Analysis for Swing Option Pricing

This script analyzes the swing option evaluation data to calculate:
1. Total option value across all simulation paths
2. Value per right
3. Exercise efficiency metrics
4. Risk metrics and statistics
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class SwingOptionAnalyzer:
    def __init__(self, csv_path, params_path):
        """Initialize the analyzer with data paths"""
        self.csv_path = csv_path
        self.params_path = params_path
        
        # Load parameters
        with open(params_path, 'r') as f:
            self.params = json.load(f)
            
        # Load evaluation data
        self.data = pd.read_csv(csv_path)
        
        print(f"Loaded {len(self.data)} observations across {self.params['n_paths_eval']} paths")
        print(f"Swing Option Parameters:")
        print(f"  - Total Rights (n_rights): {self.params['n_rights']}")
        print(f"  - Strike Price: {self.params['strike']}")
        print(f"  - Maturity: {self.params['maturity']} years")
        print(f"  - Initial Spot: {self.params['S0']}")
        print(f"  - Risk-free Rate: {self.params['risk_free_rate']}")
        
    def calculate_option_values(self):
        """Calculate swing option values and exercise statistics"""
        
        # Group by episode to analyze each simulation path
        episodes = self.data.groupby('episode_idx')
        
        results = {
            'episode_values': [],
            'total_exercises': [],
            'final_rewards': [],
            'exercise_efficiency': [],
            'average_exercise_price': [],
            'unused_rights': []
        }
        
        for episode_idx, episode_data in episodes:
            # Calculate total value for this episode (sum of all rewards)
            total_value = episode_data['reward'].sum()
            
            # Calculate total quantity exercised
            final_row = episode_data.iloc[-1]
            total_exercised = final_row['q_exerc']
            unused_rights = self.params['n_rights'] - total_exercised
            
            # Calculate exercise efficiency (% of rights used)
            efficiency = total_exercised / self.params['n_rights'] * 100
            
            # Calculate average exercise price (weighted by quantity)
            exercise_data = episode_data[episode_data['q_actual'] > 0]
            if len(exercise_data) > 0:
                avg_exercise_price = (exercise_data['spot'] * exercise_data['q_actual']).sum() / exercise_data['q_actual'].sum()
            else:
                avg_exercise_price = 0
            
            results['episode_values'].append(total_value)
            results['total_exercises'].append(total_exercised)
            results['final_rewards'].append(total_value)
            results['exercise_efficiency'].append(efficiency)
            results['average_exercise_price'].append(avg_exercise_price)
            results['unused_rights'].append(unused_rights)
        
        return results
    
    def calculate_comprehensive_metrics(self, results):
        """Calculate comprehensive valuation metrics"""
        
        episode_values = np.array(results['episode_values'])
        
        metrics = {
            # Core Valuation Metrics
            'mean_option_value': np.mean(episode_values),
            'median_option_value': np.median(episode_values),
            'std_option_value': np.std(episode_values),
            'value_per_right': np.mean(episode_values) / self.params['n_rights'],
            
            # Risk Metrics
            'value_at_risk_95': np.percentile(episode_values, 5),  # 95% VaR
            'value_at_risk_99': np.percentile(episode_values, 1),  # 99% VaR
            'expected_shortfall_95': np.mean(episode_values[episode_values <= np.percentile(episode_values, 5)]),
            
            # Exercise Metrics
            'mean_exercise_efficiency': np.mean(results['exercise_efficiency']),
            'mean_unused_rights': np.mean(results['unused_rights']),
            'mean_total_exercised': np.mean(results['total_exercises']),
            'mean_avg_exercise_price': np.mean(results['average_exercise_price']),
            
            # Distribution Metrics
            'min_value': np.min(episode_values),
            'max_value': np.max(episode_values),
            'skewness': self._calculate_skewness(episode_values),
            'kurtosis': self._calculate_kurtosis(episode_values),
            
            # Confidence Intervals
            'ci_95_lower': np.percentile(episode_values, 2.5),
            'ci_95_upper': np.percentile(episode_values, 97.5),
            'ci_99_lower': np.percentile(episode_values, 0.5),
            'ci_99_upper': np.percentile(episode_values, 99.5),
        }
        
        return metrics, episode_values
    
    def _calculate_skewness(self, data):
        """Calculate skewness of the distribution"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of the distribution"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def generate_report(self):
        """Generate comprehensive valuation report"""
        
        print("\n" + "="*80)
        print("SWING OPTION VALUATION ANALYSIS")
        print("="*80)
        
        # Calculate option values
        results = self.calculate_option_values()
        metrics, episode_values = self.calculate_comprehensive_metrics(results)
        
        print(f"\nCORE VALUATION METRICS:")
        print(f"{'Mean Option Value:':<30} {metrics['mean_option_value']:>15.6f}")
        print(f"{'Median Option Value:':<30} {metrics['median_option_value']:>15.6f}")
        print(f"{'Standard Deviation:':<30} {metrics['std_option_value']:>15.6f}")
        print(f"{'Value per Right:':<30} {metrics['value_per_right']:>15.6f}")
        
        print(f"\nRISK METRICS:")
        print(f"{'95% Value at Risk:':<30} {metrics['value_at_risk_95']:>15.6f}")
        print(f"{'99% Value at Risk:':<30} {metrics['value_at_risk_99']:>15.6f}")
        print(f"{'95% Expected Shortfall:':<30} {metrics['expected_shortfall_95']:>15.6f}")
        
        print(f"\nEXERCISE ANALYSIS:")
        print(f"{'Mean Exercise Efficiency:':<30} {metrics['mean_exercise_efficiency']:>14.2f}%")
        print(f"{'Mean Rights Exercised:':<30} {metrics['mean_total_exercised']:>15.2f}")
        print(f"{'Mean Unused Rights:':<30} {metrics['mean_unused_rights']:>15.2f}")
        print(f"{'Mean Exercise Price:':<30} {metrics['mean_avg_exercise_price']:>15.6f}")
        
        print(f"\nDISTRIBUTION CHARACTERISTICS:")
        print(f"{'Minimum Value:':<30} {metrics['min_value']:>15.6f}")
        print(f"{'Maximum Value:':<30} {metrics['max_value']:>15.6f}")
        print(f"{'Skewness:':<30} {metrics['skewness']:>15.6f}")
        print(f"{'Kurtosis:':<30} {metrics['kurtosis']:>15.6f}")
        
        print(f"\nCONFIDENCE INTERVALS:")
        print(f"{'95% CI:':<30} [{metrics['ci_95_lower']:>10.6f}, {metrics['ci_95_upper']:>10.6f}]")
        print(f"{'99% CI:':<30} [{metrics['ci_99_lower']:>10.6f}, {metrics['ci_99_upper']:>10.6f}]")
        
        print(f"\nCONTRACT SPECIFICATIONS:")
        print(f"{'Total Rights Available:':<30} {self.params['n_rights']:>15.0f}")
        print(f"{'Strike Price:':<30} {self.params['strike']:>15.2f}")
        print(f"{'Contract Maturity:':<30} {self.params['maturity']:>15.2f} years")
        print(f"{'Initial Spot Price:':<30} {self.params['S0']:>15.2f}")
        print(f"{'Risk-Free Rate:':<30} {self.params['risk_free_rate']:>14.2f}%")
        
        # Economic Interpretation
        print(f"\nECONOMIC INTERPRETATION:")
        print(f"The swing option portfolio shows a mean value of {metrics['mean_option_value']:.6f}")
        print(f"with each right being worth approximately {metrics['value_per_right']:.6f} on average.")
        print(f"The option holder exercises {metrics['mean_exercise_efficiency']:.1f}% of available rights,")
        print(f"leaving {metrics['mean_unused_rights']:.1f} rights unused on average.")
        
        if metrics['skewness'] > 0:
            skew_desc = "positively skewed (right tail)"
        elif metrics['skewness'] < 0:
            skew_desc = "negatively skewed (left tail)"
        else:
            skew_desc = "approximately symmetric"
            
        print(f"The value distribution is {skew_desc} with excess kurtosis of {metrics['kurtosis']:.3f}.")
        
        return metrics, results, episode_values
    
    def create_visualizations(self, metrics, results, episode_values):
        """Create visualization plots for the analysis"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Value Distribution Histogram
        ax1.hist(episode_values, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(metrics['mean_option_value'], color='red', linestyle='--', 
                   label=f'Mean: {metrics["mean_option_value"]:.3f}')
        ax1.axvline(metrics['median_option_value'], color='orange', linestyle='--', 
                   label=f'Median: {metrics["median_option_value"]:.3f}')
        ax1.set_xlabel('Option Value')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of Swing Option Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Exercise Efficiency Distribution
        ax2.hist(results['exercise_efficiency'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(metrics['mean_exercise_efficiency'], color='red', linestyle='--', 
                   label=f'Mean: {metrics["mean_exercise_efficiency"]:.1f}%')
        ax2.set_xlabel('Exercise Efficiency (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Exercise Efficiency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Value vs Exercise Efficiency Scatter
        ax3.scatter(results['exercise_efficiency'], episode_values, alpha=0.6, color='purple')
        ax3.set_xlabel('Exercise Efficiency (%)')
        ax3.set_ylabel('Option Value')
        ax3.set_title('Option Value vs Exercise Efficiency')
        ax3.grid(True, alpha=0.3)
        
        # Calculate correlation
        correlation = np.corrcoef(results['exercise_efficiency'], episode_values)[0, 1]
        ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax3.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # 4. Q-Q Plot for normality assessment
        from scipy import stats
        stats.probplot(episode_values, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot: Value Distribution vs Normal')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/alexanderithakis/Documents/GitHub/D4PG-QR-FRM/swing_option_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

def main():
    """Main analysis function"""
    
    # File paths
    csv_path = "/Users/alexanderithakis/Documents/GitHub/D4PG-QR-FRM/logs/Yearly_Swing/evaluation_runs/eval_run_8192.csv"
    params_path = "/Users/alexanderithakis/Documents/GitHub/D4PG-QR-FRM/logs/Yearly_Swing/Yearly_Swing_parameters.json"
    
    # Initialize analyzer
    analyzer = SwingOptionAnalyzer(csv_path, params_path)
    
    # Generate comprehensive report
    metrics, results, episode_values = analyzer.generate_report()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    analyzer.create_visualizations(metrics, results, episode_values)
    
    # Save detailed results
    detailed_results = {
        'metrics': metrics,
        'parameters': analyzer.params,
        'summary_statistics': {
            'total_simulations': len(episode_values),
            'total_observations': len(analyzer.data),
            'analysis_date': pd.Timestamp.now().isoformat()
        }
    }
    
    output_path = "/Users/alexanderithakis/Documents/GitHub/D4PG-QR-FRM/swing_option_valuation_results.json"
    with open(output_path, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_path}")
    print(f"Visualization saved to: swing_option_analysis.png")
    
    return metrics, results

if __name__ == "__main__":
    metrics, results = main()
