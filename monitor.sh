#!/bin/bash

# E2E-DRO Monitoring Script
# Run this in a separate terminal to monitor progress

echo "🔍 E2E-DRO Progress Monitor"
echo "=========================="
echo ""

# Function to show process status
show_status() {
    echo "📊 Process Status:"
    ps aux | grep "python main.py" | grep -v grep | while read line; do
        echo "   $line"
    done
    echo ""
}

# Function to show recent output
show_output() {
    echo "📝 Recent Output (last 20 lines):"
    echo "--------------------------------"
    tail -20 output.log
    echo ""
}

# Function to show progress summary
show_progress() {
    echo "🎯 Progress Summary:"
    echo "-------------------"
    
    # Count completed models
    completed=0
    if [ -f "cache/ew_net.pkl" ]; then
        echo "   ✅ ew_net: Complete"
        ((completed++))
    else
        echo "   ⏳ ew_net: Pending"
    fi
    
    if [ -f "cache/po_net.pkl" ]; then
        echo "   ✅ po_net: Complete"
        ((completed++))
    else
        echo "   ⏳ po_net: Pending"
    fi
    
    if [ -f "cache/base_net.pkl" ]; then
        echo "   ✅ base_net: Complete"
        ((completed++))
    else
        echo "   ⏳ base_net: Pending"
    fi
    
    if [ -f "cache/nom_net.pkl" ]; then
        echo "   ✅ nom_net: Complete"
        ((completed++))
    else
        echo "   ⏳ nom_net: Pending"
    fi
    
    if [ -f "cache/dr_net.pkl" ]; then
        echo "   ✅ dr_net: Complete"
        ((completed++))
    else
        echo "   ⏳ dr_net: Pending"
    fi
    
    echo ""
    echo "   📈 Overall Progress: $completed/5 models complete"
    echo ""
}

# Main monitoring loop
while true; do
    clear
    echo "🔍 E2E-DRO Progress Monitor - $(date)"
    echo "=========================================="
    echo ""
    
    show_status
    show_progress
    show_output
    
    echo "🔄 Auto-refresh every 10 seconds... (Ctrl+C to stop)"
    echo ""
    
    sleep 10
done
