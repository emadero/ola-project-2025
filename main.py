
#!/usr/bin/env python3
"""
Main entry point for the online pricing algorithms project.
This script provides a simple interface to run demonstrations of the implemented algorithms.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_experiment_1():
    """Run Experiment 1: Single product in stochastic environment"""
    print("\n" + "="*60)
    print(" Running Experiment 1: Single product in stochastic environment")
    print("   Algorithms: UCB1 (with and without inventory constraints)")
    print("="*60)
    
    try:
        from experiments import requirement1
        requirement1.main()
        print(" Experiment 1 completed successfully!")
    except ImportError:
        print("  Experiment 1 not yet implemented")
        print("   Please implement experiments/requirement1.py")
    except Exception as e:
        print(f" Error in Experiment 1: {e}")


def run_experiment_2():
    """Run Experiment 2: Multiple products in stochastic environment"""
    print("\n" + "="*60)
    print(" Running Experiment 2: Multiple products in stochastic environment")
    print("   Algorithm: Combinatorial UCB1")
    print("="*60)
    
    try:
        from experiments import requirement2
        requirement2.main()
        print(" Experiment 2 completed successfully!")
    except ImportError:
        print("  Experiment 2 not yet implemented")
        print("   Please implement experiments/requirement2.py")
    except Exception as e:
        print(f" Error in Experiment 2: {e}")


def run_experiment_3():
    """Run Experiment 3: Best-of-both-worlds with single product"""
    print("\n" + "="*60)
    print(" Running Experiment 3: Best-of-both-worlds with single product")
    print("   Algorithm: Primal-dual method with inventory constraints")
    print("="*60)
    
    try:
        from experiments import requirement3
        requirement3.main()
        print(" Experiment 3 completed successfully!")
    except ImportError:
        print("  Experiment 3 not yet implemented")
        print("   Please implement experiments/requirement3.py")
    except Exception as e:
        print(f" Error in Experiment 3: {e}")


def run_experiment_4():
    """Run Experiment 4: Best-of-both-worlds with multiple products"""
    print("\n" + "="*60)
    print(" Running Experiment 4: Best-of-both-worlds with multiple products")
    print("   Algorithm: Primal-dual method for multiple products")
    print("="*60)
    
    try:
        from experiments import requirement4
        requirement4.main()
        print(" Experiment 4 completed successfully!")
    except ImportError:
        print("  Experiment 4 not yet implemented")
        print("   Please implement experiments/requirement4.py")
    except Exception as e:
        print(f" Error in Experiment 4: {e}")


def run_experiment_5():
    """Run Experiment 5: Slightly non-stationary environments"""
    print("\n" + "="*60)
    print(" Running Experiment 5: Slightly non-stationary environments")
    print("   Algorithms: Sliding window UCB vs Primal-dual comparison")
    print("="*60)
    
    try:
        from experiments import requirement5
        requirement5.main()
        print(" Experiment 5 completed successfully!")
    except ImportError:
        print("  Experiment 5 not yet implemented")
        print("   Please implement experiments/requirement5.py")
    except Exception as e:
        print(f" Error in Experiment 5: {e}")


def main():
    """Main interactive interface"""
    print("="*80)
    print(" Online Learning for Product Pricing with Production Constraints")
    print("="*80)
    
    # Create results directories if they don't exist
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    while True:
        print("\nChoose an experiment to run:")
        print("1. Single product in stochastic environment")
        print("2. Multiple products in stochastic environment")
        print("3. Best-of-both-worlds with single product")
        print("4. Best-of-both-worlds with multiple products")
        print("5. Slightly non-stationary environments")
        print("6. Run all experiments")
        print("0. Exit")
        
        try:
            choice = input("\nEnter your choice (0-6): ").strip()
            
            if choice == "0":
                print("\n Goodbye!")
                break
            elif choice == "1":
                run_experiment_1()
            elif choice == "2":
                run_experiment_2()
            elif choice == "3":
                run_experiment_3()
            elif choice == "4":
                run_experiment_4()
            elif choice == "5":
                run_experiment_5()
            elif choice == "6":
                print("\n Running all experiments...")
                experiments = [
                    run_experiment_1,
                    run_experiment_2,
                    run_experiment_3,
                    run_experiment_4,
                    run_experiment_5
                ]
                for exp in experiments:
                    exp()
                
                print("\n" + "="*60)
                print(" All experiments completed!")
                print(" Check results/ directory for outputs")
                print("="*60)
            else:
                print(" Invalid choice. Please enter a number between 0 and 6.")
                
        except KeyboardInterrupt:
            print("\n\n Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f" Unexpected error: {e}")
            
        # Ask if user wants to continue
        if choice in ["1", "2", "3", "4", "5", "6"]:
            input("\nPress Enter to continue...")
    
    return


if __name__ == "__main__":
    main()