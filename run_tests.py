import sys
import traceback

from tests import test_nstep_replay as t


def main():
    tests = [
        t.test_circular_nstep_buffer_gamma1_n2_terminal_inside,
        t.test_circular_nstep_buffer_gamma09_n3_no_terminal,
        t.test_uniform_replay_integration,
        t.test_per_integration,
    ]
    failures = 0
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            print(f"PASS: {name}")
        except Exception:
            failures += 1
            print(f"FAIL: {name}")
            traceback.print_exc()
    if failures:
        print(f"\n{failures} test(s) failed.")
        sys.exit(1)
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
