import importlib
import os
import sys

TEST_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

def run_test(module_name, test_func_name):
    try:
        module = importlib.import_module(module_name)
        test_func = getattr(module, test_func_name)
        result = test_func()
        # Define pass/fail criteria: result is not None and not an Exception
        passed = result is not None
        print(f"[PASS] {module_name}.{test_func_name}") if passed else print(f"[FAIL] {module_name}.{test_func_name}")
        return passed
    except Exception as e:
        print(f"[ERROR] {module_name}.{test_func_name}: {e}")
        return False

def main():
    print("Running all tests in the test folder...")
    test_files = [f for f in os.listdir(TEST_DIR) if f.startswith('test_') and f.endswith('.py')]
    results = {}
    for test_file in test_files:
        module_name = f"test.{test_file[:-3]}" if TEST_DIR != '' else test_file[:-3]
        test_func_name = [name for name in dir(importlib.import_module(module_name)) if name.startswith('test_')][0]
        print(f"\n--- Running {test_file} ---")
        passed = run_test(module_name, test_func_name)
        results[test_file] = passed
    print("\nSummary:")
    for test_file, passed in results.items():
        print(f"{test_file}: {'PASS' if passed else 'FAIL'}")

if __name__ == "__main__":
    main()
