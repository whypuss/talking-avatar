#!/usr/bin/env python3
"""Backward-compatible wrapper that delegates to whytalkv3.py."""

from whytalkv3 import main

if __name__ == "__main__":
    print("[whytalk] Notice: whytalkv2.py now forwards to whytalkv3.py")
    main()
