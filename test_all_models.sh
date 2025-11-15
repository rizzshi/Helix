#!/bin/bash
echo "Testing Algorzen Helix - All Models"
echo "===================================="
echo ""
echo "1. Testing BASELINE model..."
python main.py --input data/sample_kpi_history.csv --kpi revenue --horizon 14 --model baseline --output reports/test_baseline.pdf > /dev/null 2>&1
if [ -f reports/test_baseline.pdf ]; then echo "✓ Baseline model success"; else echo "✗ Baseline failed"; fi
echo ""
echo "2. Testing GBM model..."
python main.py --input data/sample_kpi_history.csv --kpi revenue --horizon 14 --model gbm --output reports/test_gbm.pdf > /dev/null 2>&1
if [ -f reports/test_gbm.pdf ]; then echo "✓ GBM model success"; else echo "✗ GBM failed"; fi
echo ""
echo "All tests complete!"
echo "Reports generated in reports/ directory"
